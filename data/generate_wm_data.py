"""
Orchestrator: generate action-conditioned (G, s_t, a_t, s_{t + 1}, R)
transition data for IM and write it as JSONL + a graph store
"""

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from wm_actions import (
    SPINE_ALGORITHMS,
    counterfactual_actions,
    sample_injection,
    select_seeds,
)
from wm_graphs import (
    REAL_DIRECTED,
    GraphBundle,
    make_real_bundle,
    make_synthetic_bundle,
)
from wm_simulator import VALID_ACTION_OPS, ActionOp, Simulator, State

SYNTHETIC_FAMILIES = ("er", "ba", "ws", "karate")


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
    """Build one transition record for the JSONL sotrage."""
    record = {
        "graph_id": graph_id,
        "diffusion_model": diffusion_model,
        "episode_id": episode_id,
        "algorithm": algorithm,
        "branch": branch,
        "t": int(t),
        "state": state.to_dict(),
        "action": [a.to_dict() for a in action],
        "next_state": next_state.to_dict(),
        "reward": float(reward),
    }

    # Sparse {node: prob} soft targets (string keys for JSON); absent => legacy binary
    if next_marginal_infected is not None:
        record["next_marginal_infected"] = {
            str(v): round(p, 6) for v, p in next_marginal_infected.items()
        }
        record["next_marginal_frontier"] = {
            str(v): round(p, 6) for v, p in (next_marginal_frontier or {}).items()
        }

    return record


class GraphStore:
    """Saves each graph once as <graph_id>.npz + a graphs_index.json."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.graphs_dir = self.out_dir / "graphs"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
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
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._handles = {}

    def write(self, rec: dict, model: str, split: str) -> None:
        handle = self._handles.get((model, split))
        if handle is None:
            handle = open(
                self.out_dir / f"transitions_{model}_{split}.jsonl",
                "w",
                encoding="utf-8",
            )
            self._handles[(model, split)] = handle

        handle.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        for h in self._handles.values():
            h.close()

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
    ba_m: int = 3
    ws_k: int = 6
    ws_p: float = 0.1


def _iter_bundles(config: GenConfig) -> Iterator[GraphBundle]:
    if config.dataset in SYNTHETIC_FAMILIES:
        for i in range(config.num_graphs):
            yield make_synthetic_bundle(
                config.dataset,
                index=i,
                n=config.syn_nodes,
                er_p=config.er_p,
                ba_m=config.ba_m,
                ws_k=config.ws_k,
                ws_p=config.ws_p,
                seed=config.seed,
                prob_model=config.prob_model,
                uniform_p=config.uniform_p,
            )
    else:
        yield make_real_bundle(
            config.dataset, prob_model=config.prob_model, uniform_p=config.uniform_p
        )


def _resolve_k(config: GenConfig, n: int) -> int:
    if config.budget_pct is not None:
        return max(1, round(n * config.budget_pct / 100))

    return config.budget


def _assign_split(rng: np.random.Generator, split: tuple) -> str:
    r = rng.random()

    if r < split[0]:
        return "train"
    if r < split[0] + split[1]:
        return "val"

    return "test"


def _episode_transitions(
    bundle: GraphBundle,
    model: str,
    algorithm: str,
    k: int,
    rollout: int,
    config: GenConfig,
    base_rng: np.random.Generator,
    writer: TransitionWriter,
    split: str,
) -> None:
    episode_id = f"{bundle.graph_id}|{model}|{algorithm}|k{k}|r{rollout}"

    # Per-episode RNGs derived from the base seed for reproducibility.
    sel_rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
    inj_rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
    sim_seed = int(base_rng.integers(0, 2**31 - 1))

    seeds = select_seeds(bundle, k=k, algorithm=algorithm, model=model, rng=sel_rng)
    sim = Simulator(bundle.nx_graph, ic_prob_map=bundle.ic_prob_map, seed=sim_seed)
    sim.reset(model)

    s_t = State(infected=[], frontier=[])
    seed_bag = [ActionOp("add_node", v) for v in seeds]

    # Timestep loop
    for t in range(config.horizon + 1):
        # At t = 0, the action is the seed commit
        action = (
            seed_bag
            if t == 0
            else sample_injection(
                s_t,
                graph=sim.model.graph.graph,
                rng=inj_rng,
                p_inject=config.inject_p,
                action_ops=config.action_ops,
                weight_range=(config.weight_lo, config.weight_hi),
            )
        )

        # Counterfactual forks (same s_t, different a_t) at intermediate steps.
        if t > 0 and config.cf_prob > 0 and inj_rng.random() < config.cf_prob:
            snap = sim.snapshot()
            for bi, cf_bag in enumerate(
                counterfactual_actions(
                    s_t,
                    n_nodes=bundle.nx_graph.number_of_nodes(),
                    main_bag=action,
                    n=config.cf_branches,
                    rng=inj_rng,
                    action_ops=config.action_ops,
                )
            ):
                sim.restore(snap)
                s_cf, cf_inf_marg, cf_fr_marg = sim.advance_marginal(
                    cf_bag, config.mc_marginals
                )
                writer.write(
                    build_record(
                        graph_id=bundle.graph_id,
                        diffusion_model=model,
                        episode_id=episode_id,
                        algorithm=algorithm,
                        branch=f"cf_{bi}",
                        t=t,
                        state=s_t,
                        action=cf_bag,
                        next_state=s_cf,
                        reward=float(len(s_cf.infected) - len(s_t.infected)),
                        next_marginal_infected=cf_inf_marg,
                        next_marginal_frontier=cf_fr_marg,
                    ),
                    model=model,
                    split=split,
                )
            sim.restore(snap)

        # In the main transition, apply the real action, advance one step, and write the (s_t, action, s_next) record
        # The reward is the spread gain (the increase in activated-node count this step)
        s_next, inf_marg, fr_marg = sim.advance_marginal(action, config.mc_marginals)
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
                next_marginal_infected=inf_marg,
                next_marginal_frontier=fr_marg,
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
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gs = GraphStore(out_dir)
    base_rng = np.random.default_rng(config.seed)

    n_episodes = 0
    with TransitionWriter(out_dir) as writer:
        for bundle in _iter_bundles(config):
            gs.save(bundle)
            k = _resolve_k(config, bundle.nx_graph.number_of_nodes())

            for model in config.models:
                for algorithm in config.algorithms:
                    for rollout in range(config.rollouts):
                        split = _assign_split(base_rng, config.split)
                        _episode_transitions(
                            bundle,
                            model,
                            algorithm,
                            k,
                            rollout,
                            config,
                            base_rng,
                            writer,
                            split,
                        )
                        n_episodes += 1

        gs.flush()

    metadata = {
        "task": "IM_world_model_transitions",
        "config": config.__dict__,
        "n_episodes": n_episodes,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"[done] {n_episodes} episodes -> {out_dir}")

    return metadata


def parse_args() -> GenConfig:
    parser = argparse.ArgumentParser(
        description="Generate action-conditioned WM (IM) data"
    )
    parser.add_argument(
        "--dataset",
        default="cora_ml",
        choices=list(REAL_DIRECTED) + list(SYNTHETIC_FAMILIES),
    )
    parser.add_argument("--num-graphs", type=int, default=1)
    parser.add_argument("--syn-nodes", type=int, default=100)
    parser.add_argument("--er-p", type=float, default=0.05)
    parser.add_argument("--ba-m", type=int, default=3)
    parser.add_argument("--ws-k", type=int, default=6)
    parser.add_argument("--ws-p", type=float, default=0.1)
    parser.add_argument(
        "--models", nargs="+", default=["IC", "LT"], choices=["IC", "LT"]
    )
    parser.add_argument(
        "--prob-model", default="weighted", choices=["weighted", "uniform"]
    )
    parser.add_argument("--uniform-p", type=float, default=0.1)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--budget-pct", type=float, default=None)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(SPINE_ALGORITHMS),
        choices=list(SPINE_ALGORITHMS),
    )
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--inject-p", type=float, default=0.3)
    parser.add_argument(
        "--action-ops",
        nargs="*",
        default=[],
        choices=list(VALID_ACTION_OPS),
        help="ops to inject; empty = diffusion-only (no action interventions)",
    )
    parser.add_argument("--weight-lo", type=float, default=0.0)
    parser.add_argument("--weight-hi", type=float, default=1.0)
    parser.add_argument("--cf-prob", type=float, default=0.2)
    parser.add_argument("--cf-branches", type=int, default=2)
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mc-marginals",
        type=int,
        default=30,
        help="MC draws per step to estimate soft next-step marginal targets (1 = legacy single-draw binary target)",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end run (er-40, 1 graph, 2 rollouts, horizon 4)",
    )

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = str(Path(__file__).resolve().parent / "output" / args.dataset)
    if args.smoke:
        args.dataset, args.num_graphs, args.syn_nodes, args.er_p = "er", 1, 40, 0.1
        args.rollouts, args.horizon = 2, 4
        args.algorithms = ["random", "degree"]
        args.action_ops = list(VALID_ACTION_OPS)
        args.mc_marginals = 4

    return GenConfig(
        dataset=args.dataset,
        num_graphs=args.num_graphs,
        syn_nodes=args.syn_nodes,
        er_p=args.er_p,
        ba_m=args.ba_m,
        ws_k=args.ws_k,
        ws_p=args.ws_p,
        models=args.models,
        prob_model=args.prob_model,
        uniform_p=args.uniform_p,
        budget=args.budget,
        budget_pct=args.budget_pct,
        algorithms=args.algorithms,
        rollouts=args.rollouts,
        horizon=args.horizon,
        inject_p=args.inject_p,
        action_ops=args.action_ops,
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
