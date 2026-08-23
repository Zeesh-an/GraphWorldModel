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
    counterfactual_actions,
    sample_injection,
    select_seeds,
    spine_algorithms,
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

# How train/val/test are assigned.
#
#   graph_disjoint   every episode on a graph lands in the SAME split, so no
#                    graph straddles the boundary. The correct default: two
#                    episodes on one graph share its structure, its per-edge
#                    transmission probabilities and (on the main branch) its
#                    cascade, so scoring one after training on the other is
#                    leakage, and every test number it produces is optimistic by
#                    an unknown amount.
#
#   episode_random   the historical behaviour: an independent draw per episode.
#                    Kept because every checkpoint produced before 2026-08-22 was
#                    trained under it, and a comparison against those numbers has
#                    to be able to reproduce their split. It is legacy, not an
#                    alternative — nothing new should be generated with it.
#   eval_only        every graph goes to `test` and nothing to train/val. For an
#                    OOD TARGET distribution: the dataset exists to be scored by a
#                    model trained somewhere else, and giving it a train split
#                    would invite exactly the accident it is built to rule out.
#                    Also the only mode that works with a single graph, which is
#                    what a real-graph transfer set is.
graph_disjoint_split = "graph_disjoint"
episode_random_split = "episode_random"
eval_only_split = "eval_only"
valid_split_modes = (graph_disjoint_split, episode_random_split, eval_only_split)

# Below this many graphs a disjoint split cannot honour three non-empty parts,
# and a "split" that puts everything in train is worse than a loud failure.
min_graphs_for_disjoint = 3


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
    # How train/val/test are assigned; see `valid_split_modes` above. Recorded in
    # metadata.json for the same reason remove_semantics is: a number produced
    # under a leaky split is not comparable to one produced under a clean one,
    # and the difference must not be reconstructible only from memory.
    split_mode: str = graph_disjoint_split
    ba_m: int = 3
    ws_k: int = 6
    ws_p: float = 0.1
    sbm_blocks: int = 4
    sbm_p_in: float = 0.15
    sbm_p_out: float = 0.01
    # powerlaw_cluster (RL4IM) and kronecker (ConTinEst); see data/wm_graphs.py
    plc_m: int = 2
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
    """Legacy per-episode draw. Only reachable under `episode_random`."""
    draw = rng.random()

    if draw < split[0]:
        return "train"
    if draw < split[0] + split[1]:
        return "val"

    return "test"


def _graph_split_plan(
    graph_count: int, split: tuple, seed: int
) -> list[str]:
    """
    Assign each graph INDEX to a split, exactly honouring the requested ratios.

    Stratified rather than sampled: with 20 graphs an independent draw per graph
    lands a 70/15/15 split anywhere from 12 to 18 training graphs, and can empty
    `val` outright. Cutting a deterministically shuffled index list at the ratio
    boundaries makes the proportions exact and the assignment reproducible from
    `--seed` alone.

    The shuffle uses its own generator so that adding this mode does not perturb
    the `base_rng` stream that draws budgets and simulator seeds — a dataset
    regenerated with the same seed differs only in its split labels.
    """
    train_count = int(round(graph_count * split[0]))
    val_count = int(round(graph_count * split[1]))

    # Rounding can overshoot; the test split absorbs it, and train/val are
    # clamped so neither can be emptied by a rounding artefact.
    train_count = max(1, min(train_count, graph_count - 2))
    val_count = max(1, min(val_count, graph_count - train_count - 1))

    order = np.random.default_rng(seed).permutation(graph_count)
    plan = ["test"] * graph_count

    for position, index in enumerate(order):
        if position < train_count:
            plan[int(index)] = "train"
        elif position < train_count + val_count:
            plan[int(index)] = "val"

    return plan


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
            ),
            model=model,
            split=split,
        )

        # Advance s_t
        s_t = s_next
        if t > 0 and not s_t.frontier and not action:
            # Stop early when the cascade is dead (no frontier) and there are no pending injections
            break


def run_generation(config: GenConfig) -> dict[str, object]:
    generation_start = time.perf_counter()
    out_dir = Path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    graph_store = GraphStore(out_dir)
    base_rng = np.random.default_rng(config.seed)

    graph_count = 1 if config.dataset in real_directed else config.num_graphs
    total_episodes = (
        graph_count * len(config.models) * len(config.algorithms) * config.rollouts
    )

    if config.split_mode not in valid_split_modes:
        raise ValueError(
            f"unknown --split-mode {config.split_mode!r}; "
            f"choose one of {list(valid_split_modes)}"
        )

    split_plan = None

    if config.split_mode == eval_only_split:
        split_plan = ["test"] * graph_count

    if config.split_mode == graph_disjoint_split:
        if graph_count < min_graphs_for_disjoint:
            raise ValueError(
                f"--split-mode {graph_disjoint_split} needs at least "
                f"{min_graphs_for_disjoint} graphs to fill train/val/test without "
                f"leaking, but this run has {graph_count} "
                f"(dataset={config.dataset!r}). A single-graph dataset cannot be "
                f"split disjointly by graph at all.\n"
                f"  - for a synthetic family, raise --num-graphs\n"
                f"  - to reproduce a pre-2026-08-22 dataset, pass "
                f"--split-mode {episode_random_split} (leaky, legacy only)"
            )

        split_plan = _graph_split_plan(graph_count, config.split, config.seed)

    graphs_meta = []
    n_episodes = 0
    split_by_graph: dict[str, str] = {}
    progress_bar = tqdm(total=total_episodes, desc="episodes")

    with TransitionWriter(out_dir) as writer:
        for graph_index, bundle in enumerate(_iter_bundles(config)):
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
                for algorithm in config.algorithms:
                    for rollout in range(config.rollouts):
                        # The draw happens in BOTH modes and is used in only one.
                        # Keeping it unconditional holds the `base_rng` stream —
                        # and therefore every budget and simulator seed below —
                        # identical across the two modes, so regenerating a
                        # dataset with --split-mode graph_disjoint changes the
                        # split labels and nothing else. That makes the two
                        # directly comparable, which is the whole point of
                        # keeping the legacy mode around.
                        episode_split = _assign_split(base_rng, config.split)
                        split = (
                            split_plan[graph_index]
                            if split_plan is not None
                            else episode_split
                        )
                        # A SET, not a single value: under episode_random a graph
                        # legitimately carries several splits, and recording that
                        # in metadata.json is what makes an existing leaky
                        # dataset self-evident instead of a thing you have to
                        # remember.
                        split_by_graph.setdefault(bundle.graph_id, set()).add(split)
                        budget = _resolve_budget(config, num_nodes, base_rng)
                        episode_budgets.append(budget)
                        _episode_transitions(
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

    splits_per_graph = {
        graph_id: sorted(splits) for graph_id, splits in sorted(split_by_graph.items())
    }
    straddling = sorted(
        graph_id for graph_id, splits in splits_per_graph.items() if len(splits) > 1
    )

    if straddling:
        print(
            f"[warn] {len(straddling)}/{len(splits_per_graph)} graphs appear in "
            f"more than one split (split_mode={config.split_mode}). Test metrics "
            f"from this dataset are optimistic: the model sees the same graph at "
            f"train and at test time. Regenerate with --split-mode "
            f"{graph_disjoint_split} for a leakage-free split."
        )

    metadata = {
        "task": "IM_world_model_transitions",
        "config": config.__dict__,
        "n_episodes": n_episodes,
        "generation_seconds": round(generation_seconds, 1),
        "graphs": graphs_meta,
        # Provenance for the split, so a results table can state which regime it
        # was produced under without anyone reconstructing it from memory.
        "split_mode": config.split_mode,
        "splits_per_graph": splits_per_graph,
        "graphs_straddling_splits": straddling,
        "split_is_graph_disjoint": not straddling,
    }
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
        default=2,
        help="powerlaw_cluster: edges added per new node; average degree is ~2m, "
        "and RL4IM quotes 3 (default: 2).",
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
        choices=["IC", "LT"],
        help="diffusion models to generate (default: IC LT).",
    )
    parser.add_argument(
        "--prob-model",
        type=str,
        default="weighted",
        choices=["weighted", "uniform", "random"],
        help="edge probability model. `weighted` sets p(u->v) = 1/in_degree(v), "
        "`uniform` a constant, `random` an i.i.d. draw per edge. Use `random` for "
        "the hide-edge-weights ablation: under `weighted` the probability is an "
        "exact function of a node degree the model already reads as an input "
        "channel, so masking it removes nothing and the ablation is vacuous "
        "(default: weighted).",
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
        "--split-mode",
        type=str,
        default=graph_disjoint_split,
        choices=list(valid_split_modes),
        help=f"how train/val/test are assigned. {eval_only_split} puts every "
        f"graph in `test`, for an OOD target distribution scored by a model "
        f"trained elsewhere. {graph_disjoint_split} (default) "
        f"keeps every episode of a graph in one split, so no graph straddles the "
        f"boundary. {episode_random_split} draws per episode and is LEGACY: it "
        f"leaks a graph across splits and makes test metrics optimistic. Use it "
        f"only to reproduce a dataset generated before 2026-08-22.",
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
        split_mode=args.split_mode,
        weight_lo=args.weight_lo,
        weight_hi=args.weight_hi,
        cf_prob=args.cf_prob,
        cf_branches=args.cf_branches,
        split=tuple(args.split),
        seed=args.seed,
        mc_marginals=args.mc_marginals,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    run_generation(parse_args())
