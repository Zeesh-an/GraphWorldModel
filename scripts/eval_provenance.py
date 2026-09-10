"""
Experiment 2 — the intervention-provenance test.

    python -m scripts.eval_provenance \
        --data-dir experiments/ba24/data --diffusion-model IC \
        --results experiments/ba24/wm/IC_none_s42.json \
                  experiments/ba24/wm/IC_message_s42.json \
        --out experiments/ba24/provenance_IC.json

THE HYPOTHESIS, AS THE BRIEF STATES IT
--------------------------------------
    Case A: node v became active through natural diffusion.
    Case B: node v was externally selected as a new seed.
    Both have the same visible state, different provenance — so a state-only
    transition may be losing information the action carries.

THE MATCHED PAIR
----------------
Both cases are constructed to have the SAME post-intervention state, which is
what makes them a controlled comparison rather than two different states:

    A:  state = (I, F) with v in F,        action = []
    B:  state = (I - v, F - v),            action = [add_node(v)]

    T_exo(A) = (I, F) = T_exo(B)

The two differ only in where the "v is active" bit came from: in A it is in the
state columns of X, in B it is in CH_ADD.

WHAT THE SIMULATOR SAYS
-----------------------
This script first asks the SIMULATOR, not a model. NDlib's IC and LT steppers are
functions of `model.status` (plus, for LT, the per-node thresholds), so if the
post-action status dicts are byte-identical the two cases have the *same*
transition kernel and provenance carries exactly zero information about the next
step. `status_identical` reports whether that is so, and `true_divergence`
measures it empirically with matched Monte Carlo draws as a check on the
argument rather than a substitute for it.

This is the honest framing, and it can come out against the hypothesis. If the
truth does not distinguish the two cases, then a model that DOES is not
recovering provenance — it is inventing a distinction that is not there, and
`pred_divergence` measures how much of that each arm has. Lower is better, and
the exact-oracle head scores 0 by construction.

The one place provenance genuinely persists under these dynamics is LT's hidden
threshold: a node that activated naturally has revealed `theta_v <= f_v`, one
that was seeded has revealed nothing, and the difference becomes observable only
if the node is later removed back to susceptible and has to cross its threshold
again. `--threshold-probe` measures the SIZE of that channel directly, by
drawing thresholds from the two conditional distributions. No model in this
family can use it — they are all Markov in (s_t, a_t) and the distinguishing
action happened at t-1 — so it quantifies what per-step action conditioning,
feature-level or message-level, structurally cannot reach.
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from data.wm_simulator import ActionOp, Simulator
from world_model.checkpoint import load_checkpoint
from world_model.wm_data import (
    build_features,
    build_graph_input,
    load_graph_store,
)

#: Draws for the empirical matched-MC check on the true kernel.
default_mc_draws = 200


def _edges_and_probs(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(entry["edge_index"], dtype=np.int64),
        np.asarray(entry["ic_probs"], dtype=np.float32),
    )


def _nx_graph(entry: dict) -> tuple[nx.Graph, dict]:
    edge_index, probabilities = _edges_and_probs(entry)
    graph = nx.Graph()
    graph.add_nodes_from(range(int(entry["num_nodes"])))
    probability_map = {}

    for edge in range(edge_index.shape[1]):
        source, destination = int(edge_index[0, edge]), int(edge_index[1, edge])
        graph.add_edge(source, destination)
        probability_map[(source, destination)] = float(probabilities[edge])

    return graph, probability_map


def _set_status(simulator: Simulator, infected: set[int], frontier: set[int]) -> None:
    """Drive the simulator to an arbitrary (infected, frontier) state."""
    for node in simulator.model.status:
        node = int(node)

        if node in frontier:
            simulator.model.status[node] = 1
        elif node in infected:
            # IC: 2 is infected-but-spent; LT has no spent state, so an infected
            # node is simply active
            simulator.model.status[node] = 2 if simulator.model_name == "IC" else 1
        else:
            simulator.model.status[node] = 0


def _record(state: dict, action: list[dict]) -> dict:
    return {
        "state": state,
        "action": action,
        "next_state": {"infected": [], "frontier": []},
        "next_marginal_infected": {},
        "next_marginal_frontier": {},
    }


def sample_state(
    simulator: Simulator,
    num_nodes: int,
    rng: np.random.Generator,
    diffusion_model: str,
    budget: int,
    steps: int,
) -> tuple[set[int], set[int]] | None:
    """Run a short episode and return a state with a non-empty frontier."""
    simulator.reset(diffusion_model)
    seeds = rng.choice(num_nodes, size=budget, replace=False).tolist()
    state = simulator.advance([ActionOp("add_node", int(node)) for node in seeds])

    for _ in range(steps):
        if not state.frontier:
            return None

        state = simulator.advance([])

    if not state.frontier:
        return None

    return set(state.infected), set(state.frontier)


def matched_pair(
    infected: set[int], frontier: set[int], target: int
) -> tuple[dict, dict]:
    """The two records of the controlled comparison."""
    natural = _record(
        {"infected": sorted(infected), "frontier": sorted(frontier)}, []
    )
    seeded = _record(
        {
            "infected": sorted(infected - {target}),
            "frontier": sorted(frontier - {target}),
        },
        [{"op": "add_node", "target": int(target)}],
    )

    return natural, seeded


def true_kernel_divergence(
    simulator: Simulator,
    infected: set[int],
    frontier: set[int],
    target: int,
    num_nodes: int,
    draws: int,
) -> tuple[bool, float, float]:
    """
    (status dicts identical, |marginal_A - marginal_B|, the same-status noise floor).

    The boolean is the ARGUMENT — identical post-action statuses mean one shared
    kernel — and the two floats are the empirical check on it: the A-vs-B
    divergence has to be read against the floor, not against zero, because NDlib
    draws fresh randomness per call and two runs from one status already differ
    by the floor.
    """
    _set_status(simulator, infected, frontier)
    natural_status = dict(simulator.model.status)

    _set_status(simulator, infected - {target}, frontier - {target})
    simulator.apply_actions([ActionOp("add_node", int(target))])
    seeded_status = dict(simulator.model.status)

    identical = natural_status == seeded_status

    def marginal(status: dict) -> np.ndarray:
        counts = np.zeros(num_nodes)
        for _ in range(draws):
            simulator.model.status = dict(status)
            simulator.model.iteration()
            counts[sorted(simulator.active_nodes())] += 1.0

        return counts / draws

    # Three estimates, so the A-vs-B number is readable: two independent runs
    # from the SAME status give the sampling-noise floor this estimator has at
    # `draws`, and the A-vs-B difference has to be judged against it rather than
    # against zero. NDlib owns its own RNG, so identical statuses still give
    # different draws — which is exactly what the floor measures.
    natural_first = marginal(natural_status)
    natural_second = marginal(natural_status)
    seeded_marginal = marginal(seeded_status)

    return (
        identical,
        float(np.abs(natural_first - seeded_marginal).mean()),
        float(np.abs(natural_first - natural_second).mean()),
    )


@torch.inference_mode()
def model_divergence(
    model,
    spec,
    natural: dict,
    seeded: dict,
    edge_index: np.ndarray,
    weights: np.ndarray,
    num_nodes: int,
    device: torch.device,
) -> tuple[float, float]:
    """(mean, max) |P_A(infected) - P_B(infected)| over nodes."""
    predictions = []

    for record in (natural, seeded):
        features, _, _ = build_features(
            record, edge_index, num_nodes, spec.action_encoding
        )
        graph_input = build_graph_input(
            edge_index,
            weights,
            num_nodes,
            spec.diffusion_model,
            device,
            spec.hide_edge_weights,
        )
        logits = model(torch.from_numpy(features).to(device), graph_input)
        predictions.append(torch.sigmoid(logits[:, 0]).cpu().numpy())

    difference = np.abs(predictions[0] - predictions[1])

    return float(difference.mean()), float(difference.max())


def threshold_probe(
    entry: dict, rng: np.random.Generator, trials: int, budget: int, steps: int
) -> dict:
    """
    How much does provenance actually persist under LT?

    A node that activated naturally has `theta_v <= f_v`; one that was seeded has
    `theta_v ~ U(0,1)`. The two are indistinguishable while `v` stays active, so
    the probe removes `v` (LT `remove_node` returns it to susceptible) and asks
    whether it re-crosses its threshold on the next step.

        P(v active at t+2 | natural at t, removed at t+1)
        P(v active at t+2 | seeded  at t, removed at t+1)

    The gap is the whole provenance channel these dynamics contain. It is
    measured on the SIMULATOR with explicitly supplied thresholds, so it is a
    property of the world and not of any model.
    """
    graph, probability_map = _nx_graph(entry)
    num_nodes = int(entry["num_nodes"])
    simulator = Simulator(graph, probability_map, seed=0)
    natural_hits, seeded_hits, pairs = 0, 0, 0

    for trial in range(trials):
        thresholds = {
            int(node): float(rng.uniform(0.0, 1.0)) for node in graph.nodes()
        }
        simulator.reset("LT", lt_thresholds=thresholds)
        seeds = rng.choice(num_nodes, size=budget, replace=False).tolist()
        state = simulator.advance([ActionOp("add_node", int(node)) for node in seeds])

        for _ in range(steps):
            if not state.frontier:
                break
            state = simulator.advance([])

        if not state.frontier:
            continue

        target = int(rng.choice(sorted(state.frontier)))
        active = set(state.infected)
        neighbours = list(graph.neighbors(target))

        if not neighbours:
            continue

        # The active-neighbour fraction v just crossed
        fraction = sum(1 for other in neighbours if other in active) / len(neighbours)

        if fraction <= 0.0:
            continue

        pairs += 1

        for label, drawn in (
            # Natural: v activated, so its threshold is at most the fraction it
            # saw. That is the posterior a naturally-activated node induces.
            ("natural", float(rng.uniform(0.0, fraction))),
            # Seeded: the action forced it, so the threshold is untouched by the
            # activation and stays on its prior
            ("seeded", float(rng.uniform(0.0, 1.0))),
        ):
            probe_thresholds = dict(thresholds)
            probe_thresholds[target] = drawn
            simulator.reset("LT", lt_thresholds=probe_thresholds)
            _set_status(simulator, active, state.frontier)
            # t+1: hand the node back to susceptible
            simulator.advance([ActionOp("remove_node", target)])
            # t+2: does it cross again?
            after = simulator.advance([])

            if target in set(after.infected):
                if label == "natural":
                    natural_hits += 1
                else:
                    seeded_hits += 1

    if pairs == 0:
        return {"n_pairs": 0, "note": "no usable states found"}

    return {
        "n_pairs": pairs,
        "p_reactivate_natural": natural_hits / pairs,
        "p_reactivate_seeded": seeded_hits / pairs,
        "provenance_gap": (natural_hits - seeded_hits) / pairs,
        "note": (
            "the gap is the size of the multi-step provenance channel. It is "
            "invisible to any model that is Markov in (s_t, a_t): the "
            "distinguishing action happened at t-1 and is not an input at t+1"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--diffusion-model", type=str, default="IC")
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="one or more train_wm.py results JSONs; each is scored as an arm.",
    )
    parser.add_argument("--graphs", type=int, default=6)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--mc-draws", type=int, default=default_mc_draws)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--threshold-probe",
        action="store_true",
        help="also run the LT threshold-persistence probe (LT only).",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    store = load_graph_store(args.data_dir)
    graph_ids = sorted(store)[: args.graphs]

    arms = {}
    for path in args.results:
        model, spec, _ = load_checkpoint(
            Path(json.loads(path.read_text())["config"]["ckpt_dir"])
            / f"wm_{json.loads(path.read_text())['config']['model']}_"
            f"{json.loads(path.read_text())['config']['diffusion_model']}.pt",
            config=str(path),
            device=args.device,
            strict_spec=False,
        )
        arms[path.stem] = (model.eval(), spec)

    rng = np.random.default_rng(args.seed)
    per_arm = {name: {"mean": [], "max": []} for name in arms}
    identical_count, total = 0, 0
    true_divergences, noise_floors = [], []

    for graph_id in graph_ids:
        entry = store[graph_id]
        num_nodes = int(entry["num_nodes"])
        edge_index, probabilities = _edges_and_probs(entry)
        graph, probability_map = _nx_graph(entry)
        simulator = Simulator(graph, probability_map, seed=0)

        for _ in range(args.trials):
            sampled = sample_state(
                simulator,
                num_nodes,
                rng,
                args.diffusion_model,
                args.budget,
                args.steps,
            )

            if sampled is None:
                continue

            infected, frontier = sampled
            target = int(rng.choice(sorted(frontier)))
            natural, seeded = matched_pair(infected, frontier, target)

            identical, divergence, floor = true_kernel_divergence(
                simulator, infected, frontier, target, num_nodes, args.mc_draws
            )
            identical_count += int(identical)
            true_divergences.append(divergence)
            noise_floors.append(floor)
            total += 1

            for name, (model, spec) in arms.items():
                mean_difference, max_difference = model_divergence(
                    model,
                    spec,
                    natural,
                    seeded,
                    edge_index,
                    probabilities,
                    num_nodes,
                    device,
                )
                per_arm[name]["mean"].append(mean_difference)
                per_arm[name]["max"].append(max_difference)

    payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "diffusion_model": args.diffusion_model,
            "graphs": graph_ids,
            "trials_per_graph": args.trials,
            "matched_pairs": total,
            "mc_draws": args.mc_draws,
        },
        "truth": {
            "status_identical_frac": identical_count / total if total else None,
            "true_divergence_mean": float(np.mean(true_divergences)) if total else None,
            "true_divergence_max": float(np.max(true_divergences)) if total else None,
            # Two independent MC estimates from the SAME status. The A-vs-B
            # number above is meaningful only relative to this.
            "mc_noise_floor_mean": float(np.mean(noise_floors)) if total else None,
            "reading": (
                "post-action status dicts identical => the two cases share one "
                "transition kernel, so provenance carries no one-step "
                "information and any nonzero model divergence is spurious"
            ),
        },
        "arms": {
            name: {
                "pred_divergence_mean": float(np.mean(scores["mean"])),
                "pred_divergence_max": float(np.max(scores["max"])),
                "n": len(scores["mean"]),
            }
            for name, scores in per_arm.items()
            if scores["mean"]
        },
    }

    if args.threshold_probe:
        payload["threshold_persistence"] = threshold_probe(
            store[graph_ids[0]],
            np.random.default_rng(args.seed + 1),
            args.trials * len(graph_ids),
            args.budget,
            args.steps,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload["truth"], indent=2))
    for name, block in payload["arms"].items():
        print(
            f"{name}: spurious provenance divergence "
            f"mean {block['pred_divergence_mean']:.5f}, "
            f"max {block['pred_divergence_max']:.5f}"
        )

    if "threshold_persistence" in payload:
        print(json.dumps(payload["threshold_persistence"], indent=2))

    print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
