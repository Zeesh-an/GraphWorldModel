"""
Orchestrator: generate action-conditioned (G, s_t, a_t, s_{t + 1}, R)
transition data for IM and write it as JSONL + a graph store
"""

import argparse
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm

from data.wm_actions import (
    blocking_selectors,
    counterfactual_actions,
    delete_node_bag,
    immunizer_selectors as default_immunizer_selectors,
    sample_injection,
    select_blockers,
    select_immunizers,
    select_seeds,
    spine_algorithms,
)
from data.wm_competitive import (
    CompetitiveConfig,
    CompetitiveSimulator,
    auto_dominance,
    shared_positive_prob,
    tie_break_choices,
)
from data.wm_epidemic import (
    EpidemicConfig,
    EpidemicSimulator,
    default_burn_in,
)
from data.wm_graphs import (
    GraphBundle,
    kronecker_seeds,
    make_real_bundle,
    make_synthetic_bundle,
    real_directed,
)
from data.wm_simulator import (
    ActionOp,
    Simulator,
    State,
    epidemic_dynamics,
    spent,
    valid_action_ops,
    valid_remove_semantics,
)

synthetic_families = (
    "er",
    "ba",
    "ws",
    "sbm",
    "powerlaw_cluster",
    "kronecker",
    "karate",
)
seed_upper_bound = 2**31 - 1
results_root = Path("results")
default_task = "influence_maximization"

# Default k-sweep band: spans the 1%/5%/10%/20%-of-N budgets the learning-based
# IM literature reports, so one checkpoint covers the whole sweep
default_budget_pct_range = (1.0, 20.0)


# Storage
def build_record(
    graph_id: str,
    diffusion_model: str,
    episode_id: str,
    algorithm: str,
    branch: str,
    t: int,
    state: State,
    action: list[ActionOp],
    next_state: State,
    reward: float,
    next_marginal_infected: dict[int, float] | None = None,
    next_marginal_frontier: dict[int, float] | None = None,
    next_marginal_pos_infected: dict[int, float] | None = None,
    next_marginal_pos_frontier: dict[int, float] | None = None,
    negative_seeds: list[int] | None = None,
    parents: dict[int, list[int]] | None = None,
    next_marginal_incidence: dict[int, float] | None = None,
    next_marginal_exposed: dict[int, float] | None = None,
    next_marginal_infectious: dict[int, float] | None = None,
    next_marginal_recovered: dict[int, float] | None = None,
) -> dict:
    """Build one transition record for the JSONL storage."""
    record = {
        "graph_id": graph_id,
        "diffusion_model": diffusion_model,
        "episode_id": episode_id,
        "algorithm": algorithm,
        "branch": branch,
        "t": int(t),
        "state": state.to_dict(),
        "action": [action_op.to_dict() for action_op in action],
        "next_state": next_state.to_dict(),
        "reward": float(reward),
    }

    # Sparse {node: prob} soft targets (string keys for JSON); absent => legacy binary
    if next_marginal_infected is not None:
        record["next_marginal_infected"] = {
            str(node): round(probability, 6)
            for node, probability in next_marginal_infected.items()
        }
        record["next_marginal_frontier"] = {
            str(node): round(probability, 6)
            for node, probability in (next_marginal_frontier or {}).items()
        }

    # The positive cascade's two soft targets. Present only on a competitive
    # episode, so a single-cascade JSONL is byte-identical to what it was and the
    # 4-target loader can detect its own data by the presence of these keys.
    if next_marginal_pos_infected is not None:
        record["next_marginal_pos_infected"] = {
            str(node): round(probability, 6)
            for node, probability in next_marginal_pos_infected.items()
        }
        record["next_marginal_pos_frontier"] = {
            str(node): round(probability, 6)
            for node, probability in (next_marginal_pos_frontier or {}).items()
        }

    # The three CURRENT compartments plus the incidence, present only on a
    # compartmental episode so an IC/LT/competitive JSONL is byte-identical to what
    # it was. `next_marginal_infected` above stays the EVER-infected marginal under
    # this layout too, which is what lets every existing reader keep working
    # (research/epidemic_control.md §2.3).
    if next_marginal_incidence is not None:
        for key, marginal in (
            ("next_marginal_incidence", next_marginal_incidence),
            ("next_marginal_exposed", next_marginal_exposed),
            ("next_marginal_infectious", next_marginal_infectious),
            ("next_marginal_recovered", next_marginal_recovered),
        ):
            record[key] = {
                str(node): round(probability, 6)
                for node, probability in (marginal or {}).items()
            }

    # S_N is a property of the EPISODE rather than of any action (§2.1), so it is
    # stamped on every record instead of being recoverable from the t=0 bag the way
    # a seeding task's seed set is
    if negative_seeds is not None:
        record["negative_seeds"] = [int(node) for node in negative_seeds]

    # Who infected whom, for the nodes that activated at this step. Present only
    # under --trace-parents (cascade reconstruction), so every other task's JSONL
    # is byte-identical to what it was. An EMPTY list marks a node the ACTION
    # activated, i.e. a source: an add_node is an injection, not a transmission.
    if parents is not None:
        record["parents"] = {
            str(node): [int(source) for source in sources]
            for node, sources in sorted(parents.items())
        }

    return record


class GraphStore:
    """Saves each graph once as <graph_id>.npz + a graphs_index.json."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.graphs_dir = self.out_dir / "graphs"
        os.makedirs(self.graphs_dir, exist_ok=True)
        self._index = {}

    def save(self, bundle: GraphBundle) -> None:
        np.savez_compressed(
            self.graphs_dir / f"{bundle.graph_id}.npz",
            edge_index=bundle.edge_index,
            ic_probs=bundle.ic_probs,
            lt_weights=bundle.lt_weights,
            node_feats=bundle.node_feats,
            node_labels=bundle.node_labels,
        )

        self._index[bundle.graph_id] = {
            "graph_id": bundle.graph_id,
            "file": f"{bundle.graph_id}.npz",
            **bundle.meta,
        }

    def flush(self) -> None:
        (self.out_dir / "graphs_index.json").write_text(
            json.dumps(list(self._index.values()), indent=2)
        )


class TransitionWriter:
    """
    Appends transitions to transitions_<model>_<split>.jsonl, one handle per
    (model, split). "w" so a re-run regenerates cleanly rather than appending.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self._handles = {}

    def write(self, record: dict, model: str, split: str) -> None:
        handle = self._handles.get((model, split))
        if handle is None:
            handle = open(
                self.out_dir / f"transitions_{model}_{split}.jsonl",
                "w",
                encoding="utf-8",
            )
            self._handles[(model, split)] = handle

        handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()

        self._handles.clear()

    def __enter__(self) -> "TransitionWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# Generation
@dataclass
class GenConfig:
    dataset: str
    num_graphs: int
    syn_nodes: int
    er_p: float
    models: list[str]
    prob_model: str
    uniform_p: float
    budget: int
    budget_pct: float | None
    budget_pct_range: tuple[float, float] | None
    algorithms: list[str]
    rollouts: int
    horizon: int
    inject_p: float
    action_ops: list[str]
    weight_lo: float
    weight_hi: float
    cf_prob: float
    cf_branches: int
    split: tuple[float, float, float]
    seed: int
    mc_marginals: int
    out_dir: str
    # What remove_node means in this dataset; see data/wm_simulator.py. Recorded
    # in metadata.json so a checkpoint can never be trained on the wrong reading.
    remove_semantics: str = spent
    # Record the transmission edge (`parents`) on every record. Off by default
    # because NDlib never produces it and only cascade reconstruction scores it,
    # but it is a HARD PRECONDITION there: the outer-loop reward is tree-weighted
    # and a tree-weighted reward is not computable without a ground-truth parent
    # (research/cascade_reconstruction.md §2.6, §2.10 item 1).
    trace_parents: bool = False
    # Two-cascade (influence blocking) generation. `competitive` swaps the NDlib
    # Simulator for data.wm_competitive.CompetitiveSimulator and doubles the state
    # and the targets; the three below are the dynamics parameters
    # research/influence_blocking.md §8.4 and §5.1 say must be recorded rather than
    # left implicit, and they are written into metadata.json for exactly that reason.
    competitive: bool = False
    tie_break: str = auto_dominance
    positive_prob: str = shared_positive_prob
    # |S_N| as a percentage of N. §5.4 is the reason this defaults small: at
    # |S_N| = 1000 on NetHEPT even 1000 blockers remove only 17% of the negative
    # spread, so a large rumour lands every method in a regime where nothing works.
    negative_pct: float = 1.0
    negative_selectors: tuple = ("random", "degree", "pagerank")
    blocker_selectors: tuple = blocking_selectors
    # Compartmental (epidemic control) generation. `--models SIR/SIS/SEIR` swaps the
    # NDlib Simulator for data.wm_epidemic.EpidemicSimulator and widens the state
    # from two overlapping indicators to four exclusive compartments; the three
    # parameters are what research/epidemic_control.md §8.2 trap 2 says must be
    # recorded rather than left implicit, because a table that fixes beta and gamma
    # without stating them is comparable only to itself. All land in metadata.json.
    epi_beta: float = 1.0
    epi_gamma: float = 0.3
    epi_alpha: float = 0.5
    epi_burn_in: float = default_burn_in
    # |outbreak| as a percentage of N, and how each episode's dose allocation is
    # chosen. Reuses the blocker machinery: "given the outbreak's sources, pick k
    # nodes" is the same problem shape, and `none` supplies the unprotected
    # reference every prevented-infections number divides by.
    outbreak_pct: float = 1.0
    outbreak_selectors: tuple = ("random", "degree", "pagerank")
    immunizer_selectors: tuple = default_immunizer_selectors
    # Cascade prediction: REPLAY a real logged corpus instead of simulating. The one
    # flag here that changes where the data comes from at all — when it is set,
    # `run_generation` hands the whole stage to `data/wm_cascades.py` and no
    # simulator runs. research/cascade_prediction.md §9.4: these corpora are the
    # concrete, downloadable form of the "logged trajectories" our methodology note
    # asserts exist, and §2.2 is why replaying them rather than NDlib is the point.
    cascade_corpus: bool = False
    cp_observation: int = 0
    cp_horizon: int = 0
    cp_step: int = 1
    cp_min_size: int = 10
    cp_truncate: int = 100
    cp_split: str = "chronological"
    cp_target: str = "increment"
    cp_graph: str = "paths"
    cp_max_cascades: int = 0
    cp_max_nodes: int = 0
    ba_m: int = 3
    ws_k: int = 6
    ws_p: float = 0.1
    sbm_blocks: int = 4
    sbm_p_in: float = 0.15
    sbm_p_out: float = 0.01
    # powerlaw_cluster (RL4IM) and kronecker (ConTinEst); see data/wm_graphs.py
    plc_m: int = 3
    plc_p: float = 0.05
    kron_variant: str = "core_periphery"


def _iter_bundles(config: GenConfig) -> Iterator[GraphBundle]:
    if config.dataset in synthetic_families:
        for index in range(config.num_graphs):
            yield make_synthetic_bundle(
                config.dataset,
                index=index,
                num_nodes=config.syn_nodes,
                er_p=config.er_p,
                ba_m=config.ba_m,
                ws_k=config.ws_k,
                ws_p=config.ws_p,
                sbm_blocks=config.sbm_blocks,
                sbm_p_in=config.sbm_p_in,
                sbm_p_out=config.sbm_p_out,
                plc_m=config.plc_m,
                plc_p=config.plc_p,
                kron_variant=config.kron_variant,
                seed=config.seed,
                prob_model=config.prob_model,
                uniform_p=config.uniform_p,
            )
    else:
        yield make_real_bundle(
            config.dataset, prob_model=config.prob_model, uniform_p=config.uniform_p
        )


def _resolve_budget(config: GenConfig, num_nodes: int, rng: np.random.Generator) -> int:
    # A range draws a fresh k per episode, so one checkpoint covers a whole
    # k-sweep instead of only the single budget it was generated at
    if config.budget_pct_range is not None:
        pct = rng.uniform(*config.budget_pct_range)
        return max(1, round(num_nodes * pct / 100))

    if config.budget_pct is not None:
        return max(1, round(num_nodes * config.budget_pct / 100))

    return config.budget


def _assign_split(rng: np.random.Generator, split: tuple) -> str:
    draw = rng.random()

    if draw < split[0]:
        return "train"
    if draw < split[0] + split[1]:
        return "val"

    return "test"


def _episode_transitions(
    bundle: GraphBundle,
    model: str,
    algorithm: str,
    budget: int,
    rollout: int,
    config: GenConfig,
    base_rng: np.random.Generator,
    writer: TransitionWriter,
    split: str,
) -> None:
    episode_id = f"{bundle.graph_id}|{model}|{algorithm}|k{budget}|r{rollout}"

    # Per-episode RNGs derived from the base seed for reproducibility.
    selection_rng = np.random.default_rng(base_rng.integers(0, seed_upper_bound))
    injection_rng = np.random.default_rng(base_rng.integers(0, seed_upper_bound))
    simulator_seed = int(base_rng.integers(0, seed_upper_bound))

    seeds = select_seeds(
        bundle, num_seeds=budget, algorithm=algorithm, model=model, rng=selection_rng
    )
    simulator = Simulator(
        bundle.nx_graph,
        ic_prob_map=bundle.ic_prob_map,
        seed=simulator_seed,
        remove_semantics=config.remove_semantics,
        trace_parents=config.trace_parents,
    )
    simulator.reset(model)

    s_t = State(infected=[], frontier=[])
    seed_bag = [ActionOp("add_node", node) for node in seeds]

    # Timestep loop
    for t in range(config.horizon + 1):
        # At t = 0, the action is the seed commit
        action = (
            seed_bag
            if t == 0
            else sample_injection(
                s_t,
                graph=simulator.model.graph.graph,
                rng=injection_rng,
                p_inject=config.inject_p,
                action_ops=config.action_ops,
                weight_range=(config.weight_lo, config.weight_hi),
                remove_semantics=config.remove_semantics,
            )
        )

        # Counterfactual forks (same s_t, different a_t) at intermediate steps.
        if t > 0 and config.cf_prob > 0 and injection_rng.random() < config.cf_prob:
            snapshot = simulator.snapshot()
            cf_bags = counterfactual_actions(
                s_t,
                graph=simulator.model.graph.graph,
                main_bag=action,
                count=config.cf_branches,
                rng=injection_rng,
                action_ops=config.action_ops,
                remove_semantics=config.remove_semantics,
            )
            for branch_index, cf_bag in enumerate(cf_bags):
                simulator.restore(snapshot)
                s_cf, cf_infected_marginal, cf_frontier_marginal = (
                    simulator.advance_marginal(cf_bag, config.mc_marginals)
                )
                # A blocked-removal fork is a node-DELETION bag, so it stripped
                # arcs that restore() cannot put back; without this the main
                # branch resumes on a graph the fork edited
                simulator.revert_edges(cf_bag)
                writer.write(
                    build_record(
                        graph_id=bundle.graph_id,
                        diffusion_model=model,
                        episode_id=episode_id,
                        algorithm=algorithm,
                        branch=f"cf_{branch_index}",
                        t=t,
                        state=s_t,
                        action=cf_bag,
                        next_state=s_cf,
                        reward=float(len(s_cf.infected) - len(s_t.infected)),
                        next_marginal_infected=cf_infected_marginal,
                        next_marginal_frontier=cf_frontier_marginal,
                        parents=(
                            simulator.last_parents if config.trace_parents else None
                        ),
                    ),
                    model=model,
                    split=split,
                )
            simulator.restore(snapshot)

        # In the main transition, apply the real action, advance one step, and write the (s_t, action, s_next) record
        # Reading the graph AFTER the forks reverted theirs, so the main branch's
        # deletion bag is built against the adjacency it actually runs on
        # The reward is the spread gain (the increase in activated-node count this step)
        s_next, infected_marginal, frontier_marginal = simulator.advance_marginal(
            action, config.mc_marginals
        )
        writer.write(
            build_record(
                graph_id=bundle.graph_id,
                diffusion_model=model,
                episode_id=episode_id,
                algorithm=algorithm,
                branch="main",
                t=t,
                state=s_t,
                action=action,
                next_state=s_next,
                reward=float(len(s_next.infected) - len(s_t.infected)),
                next_marginal_infected=infected_marginal,
                next_marginal_frontier=frontier_marginal,
                parents=simulator.last_parents if config.trace_parents else None,
            ),
            model=model,
            split=split,
        )

        # Advance s_t
        s_t = s_next
        if t > 0 and not s_t.frontier and not action:
            # Stop early when the cascade is dead (no frontier) and there are no pending injections
            break


def _competitive_episode_transitions(
    bundle: GraphBundle,
    model: str,
    algorithm: str,
    budget: int,
    rollout: int,
    config: GenConfig,
    base_rng: np.random.Generator,
    writer: TransitionWriter,
    split: str,
) -> None:
    """
    One two-cascade episode: commit `S_N`, let a blocker answer it, record 4 targets.

    `algorithm` is a PAIR here — "<negative selector>+<blocker selector>" — because
    a blocking transition is only labelled by both. The attacker model is the second
    experimental axis this literature has and IM does not
    (research/influence_blocking.md §8.3), and the blocker selector is what supplies
    action diversity, including the `none` arm whose episodes are the unopposed
    sigma(S_N, empty) reference every prevented-influence number divides by.
    """
    negative_algorithm, _, blocker_algorithm = algorithm.partition("+")
    episode_id = (
        f"{bundle.graph_id}|{model}|{algorithm}|k{budget}|r{rollout}"
    )

    selection_rng = np.random.default_rng(base_rng.integers(0, seed_upper_bound))
    injection_rng = np.random.default_rng(base_rng.integers(0, seed_upper_bound))
    simulator_seed = int(base_rng.integers(0, seed_upper_bound))

    num_nodes = bundle.nx_graph.number_of_nodes()
    num_negative = max(1, round(num_nodes * config.negative_pct / 100))
    negative_seeds = select_seeds(
        bundle,
        num_seeds=num_negative,
        algorithm=negative_algorithm,
        model=model,
        rng=selection_rng,
    )
    blockers = select_blockers(
        bundle, negative_seeds, budget, blocker_algorithm, selection_rng
    )

    simulator = CompetitiveSimulator(
        bundle.nx_graph,
        ic_prob_map=bundle.ic_prob_map,
        seed=simulator_seed,
        config=CompetitiveConfig(
            tie_break=config.tie_break,
            positive_prob=config.positive_prob,
            remove_semantics=config.remove_semantics,
        ),
    )
    simulator.reset(model, negative_seeds)

    # S_N is already committed at t=0 — the rumour moved first, which is the whole
    # premise (§5.4: "first mover has a clear advantage") — so s_0 is NOT empty here
    s_t = simulator.current_state()
    blocker_bag = [ActionOp("add_node", node) for node in blockers]

    for t in range(config.horizon + 1):
        action = (
            blocker_bag
            if t == 0
            else sample_injection(
                s_t,
                graph=simulator.graph,
                rng=injection_rng,
                p_inject=config.inject_p,
                action_ops=config.action_ops,
                weight_range=(config.weight_lo, config.weight_hi),
                remove_semantics=config.remove_semantics,
            )
        )

        if t > 0 and config.cf_prob > 0 and injection_rng.random() < config.cf_prob:
            snapshot = simulator.snapshot()
            cf_bags = counterfactual_actions(
                s_t,
                graph=simulator.graph,
                main_bag=action,
                count=config.cf_branches,
                rng=injection_rng,
                action_ops=config.action_ops,
                remove_semantics=config.remove_semantics,
            )
            for branch_index, cf_bag in enumerate(cf_bags):
                simulator.restore(snapshot)
                s_cf, *cf_marginals = simulator.advance_marginal(
                    cf_bag, config.mc_marginals
                )
                writer.write(
                    build_record(
                        graph_id=bundle.graph_id,
                        diffusion_model=model,
                        episode_id=episode_id,
                        algorithm=algorithm,
                        branch=f"cf_{branch_index}",
                        t=t,
                        state=s_t,
                        action=cf_bag,
                        next_state=s_cf,
                        # LOWER is better here, so the reward is the negative
                        # cascade's growth and a good action drives it to zero
                        reward=float(len(s_cf.infected) - len(s_t.infected)),
                        next_marginal_infected=cf_marginals[0],
                        next_marginal_frontier=cf_marginals[1],
                        next_marginal_pos_infected=cf_marginals[2],
                        next_marginal_pos_frontier=cf_marginals[3],
                        negative_seeds=negative_seeds,
                    ),
                    model=model,
                    split=split,
                )
            # restore() puts the edge table back as well as the status, so unlike the
            # NDlib path there is no revert_edges companion to call here
            simulator.restore(snapshot)

        s_next, *marginals = simulator.advance_marginal(action, config.mc_marginals)
        writer.write(
            build_record(
                graph_id=bundle.graph_id,
                diffusion_model=model,
                episode_id=episode_id,
                algorithm=algorithm,
                branch="main",
                t=t,
                state=s_t,
                action=action,
                next_state=s_next,
                reward=float(len(s_next.infected) - len(s_t.infected)),
                next_marginal_infected=marginals[0],
                next_marginal_frontier=marginals[1],
                next_marginal_pos_infected=marginals[2],
                next_marginal_pos_frontier=marginals[3],
                negative_seeds=negative_seeds,
            ),
            model=model,
            split=split,
        )

        s_t = s_next
        # Both cascades have to be dead: a live positive frontier with a dead
        # negative one is still changing which nodes are protected next step
        if t > 0 and not s_t.frontier and not s_t.pos_frontier and not action:
            break


def _epidemic_episode_transitions(
    bundle: GraphBundle,
    model: str,
    algorithm: str,
    budget: int,
    rollout: int,
    config: GenConfig,
    base_rng: np.random.Generator,
    writer: TransitionWriter,
    split: str,
) -> None:
    """
    One compartmental episode: seed the outbreak, let a dose allocation answer it,
    record five targets.

    `algorithm` is a PAIR here — "<outbreak selector>+<immunizer selector>" — for
    the same reason a blocking episode's is: an intervention transition is only
    labelled by both, and the outbreak model is a second experimental axis a seeding
    task does not have. The `none` immunizer leaves the outbreak unopposed and its
    episodes are the sigma(outbreak, empty) reference.

    Two things differ from the competitive path and both matter:

      * **The outbreak is an `add_node` bag at t=0**, not a `reset` argument. That
        keeps the t=0 action readable as the source set by
        `wm_data.load_episode_endpoints` exactly as every other task's is, and it is
        also what the coding-agent harness injects, so the data and the inference
        path commit the outbreak the same way.
      * **The doses ride in the SAME t=0 bag** as full deletion bags
        (`delete_node_bag`), because a vaccinated node's incident arcs have to be
        gone from `edge_index` for the head's T_exo to be right — the identical
        requirement critical node detection has, and the reason `expand_removals`
        exists on the inference side.
    """
    outbreak_algorithm, _, immunizer_algorithm = algorithm.partition("+")
    episode_id = f"{bundle.graph_id}|{model}|{algorithm}|k{budget}|r{rollout}"

    selection_rng = np.random.default_rng(base_rng.integers(0, seed_upper_bound))
    injection_rng = np.random.default_rng(base_rng.integers(0, seed_upper_bound))
    simulator_seed = int(base_rng.integers(0, seed_upper_bound))

    num_nodes = bundle.nx_graph.number_of_nodes()
    num_sources = max(1, round(num_nodes * config.outbreak_pct / 100))
    sources = select_seeds(
        bundle,
        num_seeds=num_sources,
        algorithm=outbreak_algorithm,
        model="IC",
        rng=selection_rng,
    )
    doses = select_immunizers(
        bundle, sources, budget, immunizer_algorithm, selection_rng
    )

    simulator = EpidemicSimulator(
        bundle.nx_graph,
        ic_prob_map=bundle.ic_prob_map,
        seed=simulator_seed,
        config=EpidemicConfig(
            beta_scale=config.epi_beta,
            gamma=config.epi_gamma,
            alpha=config.epi_alpha,
            remove_semantics=config.remove_semantics,
            burn_in=config.epi_burn_in,
        ),
    )
    simulator.reset(model)

    s_t = simulator.current_state()
    opening_bag = [ActionOp("add_node", node) for node in sources]
    for node in doses:
        opening_bag += delete_node_bag(bundle.nx_graph, node)

    for t in range(config.horizon + 1):
        action = (
            opening_bag
            if t == 0
            else sample_injection(
                s_t,
                graph=simulator.graph,
                rng=injection_rng,
                p_inject=config.inject_p,
                action_ops=config.action_ops,
                weight_range=(config.weight_lo, config.weight_hi),
                remove_semantics=config.remove_semantics,
            )
        )

        if t > 0 and config.cf_prob > 0 and injection_rng.random() < config.cf_prob:
            snapshot = simulator.snapshot()
            cf_bags = counterfactual_actions(
                s_t,
                graph=simulator.graph,
                main_bag=action,
                count=config.cf_branches,
                rng=injection_rng,
                action_ops=config.action_ops,
                remove_semantics=config.remove_semantics,
            )
            for branch_index, cf_bag in enumerate(cf_bags):
                simulator.restore(snapshot)
                s_cf, *cf_marginals = simulator.advance_marginal(
                    cf_bag, config.mc_marginals
                )
                writer.write(
                    build_record(
                        graph_id=bundle.graph_id,
                        diffusion_model=model,
                        episode_id=episode_id,
                        algorithm=algorithm,
                        branch=f"cf_{branch_index}",
                        t=t,
                        state=s_t,
                        action=cf_bag,
                        next_state=s_cf,
                        # LOWER is better: the reward is the attack set's growth,
                        # so a good dose drives it to zero
                        reward=float(len(s_cf.infected) - len(s_t.infected)),
                        next_marginal_infected=cf_marginals[0],
                        # `frontier` is the infectious set under this layout, so the
                        # channel every existing reader calls "frontier" is column 3
                        next_marginal_frontier=cf_marginals[3],
                        next_marginal_incidence=cf_marginals[1],
                        next_marginal_exposed=cf_marginals[2],
                        next_marginal_infectious=cf_marginals[3],
                        next_marginal_recovered=cf_marginals[4],
                    ),
                    model=model,
                    split=split,
                )
            # restore() puts the edge table back as well as the compartments, so
            # unlike the NDlib path there is no revert_edges companion to call
            simulator.restore(snapshot)

        s_next, *marginals = simulator.advance_marginal(action, config.mc_marginals)
        writer.write(
            build_record(
                graph_id=bundle.graph_id,
                diffusion_model=model,
                episode_id=episode_id,
                algorithm=algorithm,
                branch="main",
                t=t,
                state=s_t,
                action=action,
                next_state=s_next,
                reward=float(len(s_next.infected) - len(s_t.infected)),
                next_marginal_infected=marginals[0],
                next_marginal_frontier=marginals[3],
                next_marginal_incidence=marginals[1],
                next_marginal_exposed=marginals[2],
                next_marginal_infectious=marginals[3],
                next_marginal_recovered=marginals[4],
            ),
            model=model,
            split=split,
        )

        s_t = s_next
        # Both have to be dead: under SEIR a latent node with nobody infectious
        # left is still going to become infectious, so breaking on I alone would
        # truncate the epidemic mid-flight
        if t > 0 and not s_t.frontier and not s_t.exposed and not action:
            break


def _epidemic_algorithms(config: GenConfig) -> list[str]:
    """The `<outbreak>+<immunizer>` pairs one compartmental sweep rolls out."""
    return [
        f"{outbreak}+{immunizer}"
        for outbreak in config.outbreak_selectors
        for immunizer in config.immunizer_selectors
    ]


def _competitive_algorithms(config: GenConfig) -> list[str]:
    """The `<attacker>+<blocker>` pairs one competitive sweep rolls out."""
    return [
        f"{negative}+{blocker}"
        for negative in config.negative_selectors
        for blocker in config.blocker_selectors
    ]


def run_generation(config: GenConfig) -> dict[str, object]:
    # A REPLAYED corpus never touches this function's simulator loop: there is no
    # seed selector, no injection, no counterfactual fork and no MC marginal,
    # because the cascade happened once and we are reading it back. Delegating
    # rather than branching keeps the two paths honestly separate — a reader of
    # either one can see which artifacts it writes without tracing a flag through
    # 300 lines (research/cascade_prediction.md §2.4).
    if config.cascade_corpus:
        from data.wm_cascades import ReplayConfig, replay_corpus

        return replay_corpus(
            ReplayConfig(
                dataset=config.dataset,
                out_dir=config.out_dir,
                observation=config.cp_observation,
                horizon=config.cp_horizon,
                step=config.cp_step,
                gen_horizon=config.horizon,
                min_observed=config.cp_min_size,
                truncate=config.cp_truncate,
                split=config.cp_split,
                target=config.cp_target,
                graph=config.cp_graph,
                max_cascades=config.cp_max_cascades,
                max_nodes=config.cp_max_nodes,
                prob_model=config.prob_model,
                uniform_p=config.uniform_p,
                seed=config.seed,
                diffusion_models=tuple(config.models),
            )
        )

    generation_start = time.perf_counter()
    out_dir = Path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    graph_store = GraphStore(out_dir)
    base_rng = np.random.default_rng(config.seed)

    graph_count = 1 if config.dataset in real_directed else config.num_graphs
    # A compartmental sweep's "algorithm" is an (outbreak, immunizer) PAIR and a
    # competitive one an (attacker, blocker) PAIR, so in both cases the two selector
    # lists cross rather than the spine list being used at all
    compartmental = any(model in epidemic_dynamics for model in config.models)

    if compartmental and not all(model in epidemic_dynamics for model in config.models):
        raise ValueError(
            f"--models {config.models} mixes compartmental dynamics with IC/LT. They "
            f"produce different state layouts (4 exclusive compartments vs 2 "
            f"overlapping indicators) and different target widths, so one dataset "
            f"cannot hold both. Generate them into separate runs."
        )

    if compartmental and config.competitive:
        raise ValueError(
            "--competitive is a two-CASCADE layout and the compartmental models are "
            "a four-COMPARTMENT one; no head reads both"
        )

    algorithms = (
        _epidemic_algorithms(config)
        if compartmental
        else _competitive_algorithms(config)
        if config.competitive
        else config.algorithms
    )
    total_episodes = graph_count * len(config.models) * len(algorithms) * config.rollouts

    if compartmental:
        print(
            f"[gen] compartmental: models={list(config.models)} "
            f"beta_scale={config.epi_beta} gamma={config.epi_gamma} "
            f"alpha={config.epi_alpha} |outbreak|={config.outbreak_pct}% of N, "
            f"{len(algorithms)} (outbreak+immunizer) pairs"
        )

    if config.competitive:
        print(
            f"[gen] competitive: tie_break={config.tie_break} "
            f"positive_prob={config.positive_prob} "
            f"|S_N|={config.negative_pct}% of N, "
            f"{len(algorithms)} (attacker+blocker) pairs"
        )

    graphs_meta = []
    n_episodes = 0
    progress_bar = tqdm(total=total_episodes, desc="episodes")

    with TransitionWriter(out_dir) as writer:
        for bundle in _iter_bundles(config):
            graph_store.save(bundle)

            num_nodes = bundle.nx_graph.number_of_nodes()
            num_edges = bundle.nx_graph.number_of_edges()
            episode_budgets = []

            budget_label = (
                f"k ~ U({config.budget_pct_range[0]}%, "
                f"{config.budget_pct_range[1]}% of N) per episode"
                if config.budget_pct_range is not None
                else f"k={_resolve_budget(config, num_nodes, base_rng)}"
            )
            tqdm.write(
                f"[gen] {bundle.graph_id}: N={num_nodes} E={num_edges} "
                f"budget {budget_label}"
            )

            for model in config.models:
                for algorithm in algorithms:
                    for rollout in range(config.rollouts):
                        split = _assign_split(base_rng, config.split)
                        budget = _resolve_budget(config, num_nodes, base_rng)
                        episode_budgets.append(budget)
                        episode = (
                            _epidemic_episode_transitions
                            if compartmental
                            else _competitive_episode_transitions
                            if config.competitive
                            else _episode_transitions
                        )
                        episode(
                            bundle,
                            model,
                            algorithm,
                            budget,
                            rollout,
                            config,
                            base_rng,
                            writer,
                            split,
                        )
                        n_episodes += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix(graph=bundle.graph_id[:24])

            graphs_meta.append(
                {
                    "graph_id": bundle.graph_id,
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "budget_k_min": min(episode_budgets),
                    "budget_k_max": max(episode_budgets),
                    "budget_pct_min": round(
                        100.0 * min(episode_budgets) / num_nodes, 3
                    ),
                    "budget_pct_max": round(
                        100.0 * max(episode_budgets) / num_nodes, 3
                    ),
                }
            )

        graph_store.flush()

    progress_bar.close()
    generation_seconds = time.perf_counter() - generation_start

    metadata = {
        "task": "IM_world_model_transitions",
        "config": config.__dict__,
        "n_episodes": n_episodes,
        "generation_seconds": round(generation_seconds, 1),
        "graphs": graphs_meta,
    }

    # The competitive dynamics parameters, RESOLVED. `auto` is not a value anything
    # downstream can act on, and §8.4's whole point is that the tie-break is a
    # reported hyperparameter rather than an implementation detail — so what was
    # actually simulated is written per dynamics, not what was typed.
    if config.competitive:
        competitive = CompetitiveConfig(
            tie_break=config.tie_break,
            positive_prob=config.positive_prob,
            remove_semantics=config.remove_semantics,
        )
        metadata["competitive"] = {
            model: competitive.resolved(model) for model in config.models
        } | {"negative_pct": config.negative_pct}

    # ...and the compartmental ones, for the reason §8.2 trap 2 gives: beta and
    # gamma are free parameters nobody standardizes, so a table that fixes them
    # without stating them is comparable only to itself. `train_wm` reads these back
    # rather than taking them from a flag, so a head can never be fit against
    # transitions a different rate produced.
    if compartmental:
        epidemic = EpidemicConfig(
            beta_scale=config.epi_beta,
            gamma=config.epi_gamma,
            alpha=config.epi_alpha,
            remove_semantics=config.remove_semantics,
            burn_in=config.epi_burn_in,
        )
        metadata["epidemic"] = {
            model: epidemic.resolved(model) for model in config.models
        } | {"outbreak_pct": config.outbreak_pct}

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"[done] {n_episodes} episodes in {generation_seconds:.1f}s -> {out_dir}")

    return metadata


def parse_args() -> GenConfig:
    parser = argparse.ArgumentParser(
        description="Generate action-conditioned WM (IM) data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora_ml",
        choices=list(real_directed) + list(synthetic_families),
        help="dataset name (default: cora_ml).",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=1,
        help="number of synthetic graph instances (default: 1).",
    )
    parser.add_argument(
        "--syn-nodes",
        type=int,
        default=100,
        help="nodes per synthetic graph (default: 100).",
    )
    parser.add_argument(
        "--er-p",
        type=float,
        default=0.05,
        help="ER edge probability (default: 0.05).",
    )
    parser.add_argument(
        "--ba-m",
        type=int,
        default=3,
        help="BA attachment count (default: 3).",
    )
    parser.add_argument(
        "--ws-k",
        type=int,
        default=6,
        help="WS ring degree (default: 6).",
    )
    parser.add_argument(
        "--ws-p",
        type=float,
        default=0.1,
        help="WS rewire probability (default: 0.1).",
    )
    parser.add_argument(
        "--sbm-blocks",
        type=int,
        default=4,
        help="SBM community count (default: 4).",
    )
    parser.add_argument(
        "--sbm-p-in",
        type=float,
        default=0.15,
        help="SBM within-block edge probability (default: 0.15).",
    )
    parser.add_argument(
        "--sbm-p-out",
        type=float,
        default=0.01,
        help="SBM cross-block edge probability (default: 0.01).",
    )
    parser.add_argument(
        "--plc-m",
        type=int,
        default=3,
        help="powerlaw_cluster: edges added per new node. RL4IM's own config uses "
        "3 (avg degree ~5.9); its paper says avg degree 3, which no integer m "
        "produces, so the code wins (default: 3).",
    )
    parser.add_argument(
        "--plc-p",
        type=float,
        default=0.05,
        help="powerlaw_cluster: probability an attachment closes a triangle "
        "(RL4IM: 0.05) (default: 0.05).",
    )
    parser.add_argument(
        "--kron-variant",
        type=str,
        default="core_periphery",
        choices=sorted(kronecker_seeds),
        help="kronecker seed matrix, from ConTinEst (default: core_periphery).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["IC", "LT"],
        choices=["IC", "LT"] + list(epidemic_dynamics),
        help="dynamics to generate. IC/LT run the NDlib simulator and produce two "
        "overlapping state indicators; SIR/SIS/SEIR run data/wm_epidemic.py and "
        "produce four exclusive compartments, so the two families cannot share a "
        "dataset (default: IC LT).",
    )
    parser.add_argument(
        "--prob-model",
        type=str,
        default="weighted",
        choices=["weighted", "uniform"],
        help="edge probability model (default: weighted).",
    )
    parser.add_argument(
        "--uniform-p",
        type=float,
        default=0.1,
        help="uniform IC probability when --prob-model uniform (default: 0.1).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5,
        help="absolute seed budget (default: 5).",
    )
    parser.add_argument(
        "--budget-pct",
        type=float,
        default=None,
        help="seed budget as percent of nodes (default: None).",
    )
    parser.add_argument(
        "--budget-pct-range",
        type=float,
        nargs=2,
        default=default_budget_pct_range,
        metavar=("LO", "HI"),
        help=f"sample the seed budget per episode from this percent-of-nodes range; takes precedence over --budget and --budget-pct (default: {default_budget_pct_range[0]} {default_budget_pct_range[1]}).",
    )
    parser.add_argument(
        "--no-budget-range",
        action="store_true",
        help="disable per-episode budget sampling and fall back to --budget / --budget-pct (default: False).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=list(spine_algorithms),
        choices=list(spine_algorithms),
        help="spine seed selectors to roll out (default: all spine algorithms).",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=10,
        help="episodes per graph/model/algorithm (default: 10).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="maximum timesteps per episode (default: 10).",
    )
    parser.add_argument(
        "--inject-p",
        type=float,
        default=0.3,
        help="probability of intermediate action injection (default: 0.3).",
    )
    parser.add_argument(
        "--action-ops",
        type=str,
        nargs="*",
        default=[],
        choices=list(valid_action_ops),
        help="ops to inject; empty = diffusion-only (default: []).",
    )
    parser.add_argument(
        "--remove-semantics",
        type=str,
        default=spent,
        choices=list(valid_remove_semantics),
        help="what remove_node means: spent = stays counted, stops spreading "
        "(influence maximization); blocked = deleted from the graph, uncounted, "
        "cannot transmit or be infected (containment) (default: spent).",
    )
    parser.add_argument(
        "--trace-parents",
        action="store_true",
        help="record the transmission edge (which u infected v) on every record. "
        "NDlib never produces it, so this swaps in a traced IC/LT model; needed by "
        "cascade reconstruction, where the tree-weighted reward is not computable "
        "without it (default: False).",
    )
    parser.add_argument(
        "--competitive",
        action="store_true",
        help="generate TWO-cascade (influence blocking) episodes: a negative seed "
        "set is committed at t=0 and a blocker answers it, with four soft targets "
        "per step instead of two (default: False).",
    )
    parser.add_argument(
        "--tie-break",
        type=str,
        default=auto_dominance,
        choices=list(tie_break_choices),
        help="competitive only: which cascade wins a node both reach in the same "
        "step. auto = each dynamics' own founding paper, i.e. positive dominance "
        f"under IC (Budak) and negative under LT (He) (default: {auto_dominance}).",
    )
    parser.add_argument(
        "--positive-prob",
        type=str,
        default=shared_positive_prob,
        help="competitive only: the limiting campaign's per-edge transmission "
        "probability. 'shared' is COICM (one probability per edge, independent of "
        "information type); a float is MCICM, and 1.0 is Budak's high-effectiveness "
        f"property, the case his Theorem 4.2 proves submodular (default: {shared_positive_prob}).",
    )
    parser.add_argument(
        "--negative-pct",
        type=float,
        default=1.0,
        help="competitive only: |S_N| as a percentage of N. Kept small on purpose: "
        "at |S_N| = 1000 on NetHEPT even 1000 blockers remove only 17 percent of the "
        "negative spread, so a large rumour puts every method in a regime where "
        "nothing works (default: 1.0).",
    )
    parser.add_argument(
        "--negative-selectors",
        type=str,
        nargs="+",
        default=["random", "degree", "pagerank"],
        choices=list(spine_algorithms),
        help="competitive only: how S_N is chosen — the attacker model, which is a "
        "second experimental axis IM does not have (default: random degree pagerank).",
    )
    parser.add_argument(
        "--blocker-selectors",
        type=str,
        nargs="+",
        default=list(blocking_selectors),
        choices=list(blocking_selectors),
        help="competitive only: how each episode's t=0 blocker set is chosen. "
        "`none` leaves the rumour unopposed and is the sigma(S_N, empty) reference "
        f"(default: {' '.join(blocking_selectors)}).",
    )
    parser.add_argument(
        "--epi-beta",
        type=float,
        default=1.0,
        help="compartmental only: multiplier on the graph's own per-arc probability, "
        "so beta_uv = clip(scale * p(u->v)). 1.0 leaves it at the weighted-cascade "
        "value; the literature's scalar-beta regime is --prob-model uniform "
        "--uniform-p <beta> with this at 1.0 (default: 1.0).",
    )
    parser.add_argument(
        "--epi-gamma",
        type=float,
        default=0.3,
        help="compartmental only: rate of LEAVING I — recovery under SIR/SEIR, "
        "return-to-susceptible under SIS. One parameter for both because the "
        "lambda1 * beta / delta < 1 threshold uses one. 1.0 under SIR reproduces IC "
        "exactly (default: 0.3).",
    )
    parser.add_argument(
        "--epi-alpha",
        type=float,
        default=0.5,
        help="compartmental only: E -> I rate, SEIR only (default: 0.5).",
    )
    parser.add_argument(
        "--epi-burn-in",
        type=float,
        default=default_burn_in,
        help="compartmental only: fraction of the prevalence curve discarded before "
        "the endemic prevalence is time-averaged. SIS has no terminal state, so "
        f"final size is undefined there (default: {default_burn_in}).",
    )
    parser.add_argument(
        "--outbreak-pct",
        type=float,
        default=1.0,
        help="compartmental only: outbreak size as a percentage of N (default: 1.0).",
    )
    parser.add_argument(
        "--outbreak-selectors",
        type=str,
        nargs="+",
        default=["random", "degree", "pagerank"],
        choices=list(spine_algorithms),
        help="compartmental only: how each episode's index cases are chosen — the "
        "outbreak model, a second experimental axis a seeding task does not have "
        "(default: random degree pagerank).",
    )
    parser.add_argument(
        "--immunizer-selectors",
        type=str,
        nargs="+",
        default=list(default_immunizer_selectors),
        choices=list(default_immunizer_selectors),
        help="compartmental only: how each episode's t=0 dose allocation is chosen. "
        "`none` leaves the outbreak unprotected and is the sigma(outbreak, empty) "
        f"reference (default: {' '.join(default_immunizer_selectors)}).",
    )
    parser.add_argument(
        "--weight-lo",
        type=float,
        default=0.0,
        help="minimum sampled edge weight (default: 0.0).",
    )
    parser.add_argument(
        "--weight-hi",
        type=float,
        default=1.0,
        help="maximum sampled edge weight (default: 1.0).",
    )
    parser.add_argument(
        "--cf-prob",
        type=float,
        default=0.2,
        help="counterfactual fork probability (default: 0.2).",
    )
    parser.add_argument(
        "--cf-branches",
        type=int,
        default=2,
        help="counterfactual branches per fork (default: 2).",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        help="train/val/test split probabilities (default: 0.7 0.15 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="master random seed (default: 42).",
    )
    parser.add_argument(
        "--mc-marginals",
        type=int,
        default=30,
        help="MC draws per step for soft next-step marginal targets (default: 30).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, help="output directory (default: None)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=default_task,
        help="graph task, used only to place the default --out-dir "
        f"(default: {default_task}).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="default",
        help="run label, used only to place the default --out-dir "
        "(default: default).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end run (er-40, 1 graph, 2 rollouts, horizon 4) (default: False).",
    )

    args = parser.parse_args()

    # Every generated artifact lives under results/<task>/<dataset>/<run>/;
    # the pipeline passes --out-dir explicitly, this default is for standalone
    # invocations and must match pipeline.layout.Layout
    if args.out_dir is None:
        args.out_dir = str(
            results_root / args.task / args.dataset / args.run / "data"
        )

    if args.smoke:
        args.dataset, args.num_graphs, args.syn_nodes, args.er_p = "er", 1, 40, 0.1
        args.rollouts, args.horizon = 2, 4
        args.algorithms = ["random", "degree"]
        args.action_ops = list(valid_action_ops)
        args.mc_marginals = 4

    # Tree-level metrics need a transmission edge and the parents field is where
    # it lands; this is the standalone counterpart of the registry-driven default
    # `pipeline.run` applies
    if args.task == "cascade_reconstruction":
        args.trace_parents = True

    return GenConfig(
        dataset=args.dataset,
        num_graphs=args.num_graphs,
        syn_nodes=args.syn_nodes,
        er_p=args.er_p,
        ba_m=args.ba_m,
        ws_k=args.ws_k,
        ws_p=args.ws_p,
        sbm_blocks=args.sbm_blocks,
        sbm_p_in=args.sbm_p_in,
        sbm_p_out=args.sbm_p_out,
        plc_m=args.plc_m,
        plc_p=args.plc_p,
        kron_variant=args.kron_variant,
        models=args.models,
        prob_model=args.prob_model,
        uniform_p=args.uniform_p,
        budget=args.budget,
        budget_pct=args.budget_pct,
        budget_pct_range=(
            None if args.no_budget_range else tuple(args.budget_pct_range)
        ),
        algorithms=args.algorithms,
        rollouts=args.rollouts,
        horizon=args.horizon,
        inject_p=args.inject_p,
        action_ops=args.action_ops,
        remove_semantics=args.remove_semantics,
        weight_lo=args.weight_lo,
        weight_hi=args.weight_hi,
        cf_prob=args.cf_prob,
        cf_branches=args.cf_branches,
        split=tuple(args.split),
        seed=args.seed,
        mc_marginals=args.mc_marginals,
        out_dir=args.out_dir,
        trace_parents=args.trace_parents,
        competitive=args.competitive,
        tie_break=args.tie_break,
        positive_prob=args.positive_prob,
        negative_pct=args.negative_pct,
        negative_selectors=tuple(args.negative_selectors),
        blocker_selectors=tuple(args.blocker_selectors),
        epi_beta=args.epi_beta,
        epi_gamma=args.epi_gamma,
        epi_alpha=args.epi_alpha,
        epi_burn_in=args.epi_burn_in,
        outbreak_pct=args.outbreak_pct,
        outbreak_selectors=tuple(args.outbreak_selectors),
        immunizer_selectors=tuple(args.immunizer_selectors),
    )


if __name__ == "__main__":
    run_generation(parse_args())
