"""One-step eval, action-specific metrics, free-running rollout, planning."""

import json
import sys
from pathlib import Path
from collections import defaultdict
import networkx as nx
import numpy as np
import torch

from wm_data import (
    TransitionDataset,
    collate_transitions,
    build_features,
    build_graph_input,
    reconstruct_episode_adjacency,
    edges_to_arrays,
    CH_INFECTED,
    CH_FRONTIER,
)
from wm_metrics import score_predictions, persistence_baseline, binary_f1, brier_score

_ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from data.wm_simulator import Simulator, ActionOp


@torch.inference_mode()
def evaluate_one_step(
    model: torch.nn.Module,
    dataset: TransitionDataset,
    diffusion_model: str,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Teacher-forced one-step evaluation over an entire dataset.

    Aggregates the full metric suite (score_predictions), action-specific metrics
    (Add-Seed Success, Remove-Frontier Success), action sensitivity (how much the
    model output changes under counterfactual actions at the same state), and the persistence baseline.
    """
    model.eval()
    P_inf, P_fr, PR_inf, PR_fr, Y_inf, Y_fr, I_t, F_t = [], [], [], [], [], [], [], []

    # sens[state_key][action_key] = tuple of per-node predicted infected probabilities
    sens = defaultdict(dict)
    add_hits = add_tot = rem_hits = rem_tot = 0

    for i in range(len(dataset)):
        item = dataset[i]
        batch = collate_transitions([item], diffusion_model, device)
        logits = model(batch["X"], batch["graph"])  # (N, 2)
        prob = torch.sigmoid(logits).cpu().numpy()
        pi = (prob[:, 0] > threshold).astype(np.float32)
        P_inf.append(pi)
        PR_inf.append(prob[:, 0])

        pf = (prob[:, 1] > threshold).astype(np.float32)
        P_fr.append(pf)
        PR_fr.append(prob[:, 1])

        yi = item["y_inf"].numpy()
        Y_inf.append(yi)

        yf = item["y_fr"].numpy()
        Y_fr.append(yf)

        it = item["X"][:, CH_INFECTED].numpy()
        I_t.append(it)

        ft = item["X"][:, CH_FRONTIER].numpy()
        F_t.append(ft)

        r = item["record"]
        for op in r["action"]:
            if op["op"] == "add_node":
                add_tot += 1
                add_hits += int(pi[int(op["target"])] == 1)
            elif op["op"] == "remove_node":
                rem_tot += 1
                rem_hits += int(pf[int(op["target"])] == 0)

        # Group main and cf transitions by (graph, episode, t, state) for sensitivity
        skey = (r["graph_id"], r["episode_id"], r["t"], tuple(r["state"]["infected"]))
        akey = tuple(
            sorted(
                (a["op"], a["target"], a.get("destination", -1), a.get("weight", -1.0))
                for a in r["action"]
            )
        )
        sens[skey][akey] = tuple(pi.tolist())

    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)

    # Targets are soft marginals: threshold at 0.5 for the binary F1/accuracy suite,
    # keep the raw probabilities + soft targets for Brier (calibration vs the marginal).
    Yi, Yf, It, Ft = cat(Y_inf), cat(Y_fr), cat(I_t), cat(F_t)
    Yi_bin = (Yi > 0.5).astype(np.float32)
    Yf_bin = (Yf > 0.5).astype(np.float32)

    out = score_predictions(cat(P_inf), cat(P_fr), Yi_bin, Yf_bin, It, Ft)
    out["add_seed_success"] = add_hits / add_tot if add_tot else float("nan")
    out["remove_frontier_success"] = rem_hits / rem_tot if rem_tot else float("nan")

    # Action sensitivity: mean number of distinct output tuples per (state, multiple actions) group
    diffs = [len(set(d.values())) - 1 for d in sens.values() if len(d) >= 2]
    out["action_sensitivity"] = float(np.mean(diffs)) if diffs else 0.0

    out["brier_infected"] = brier_score(cat(PR_inf), Yi)
    out["brier_frontier"] = brier_score(cat(PR_fr), Yf)

    out["persistence"] = persistence_baseline(Yi_bin, Yf_bin, It, Ft)
    out["persistence"]["brier_infected"] = brier_score(It, Yi)
    out["persistence"]["brier_frontier"] = brier_score(Ft, Yf)
    return out


@torch.inference_mode()
def rollout_episodes(
    model: torch.nn.Module,
    out_dir: str,
    diffusion_model: str,
    store: dict[str, dict],
    device: torch.device,
    split: str = "test",
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Feed the model its own thresholded prediction + recorded action; compare to truth.

    For each episode (main branch only), the model is initialised with the true state
    at t = 0 and then fed its own binary output at each subsequent step.  The recorded
    (ground-truth) action is applied at every step so the model sees the correct intervention signal.

    Three metrics are reported:
    - rollout_newinf_f1: per-step F1 on newly infected nodes only (susceptible mask)
    - rollout_count_mae: per-step |predicted count - true count| of infected nodes
    - rollout_final_f1: F1 between rolled-out final state and true final state
    """
    path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    # Keep only main-branch records; group by (graph_id, episode_id).
    by_ep = defaultdict(list)
    for r in recs:
        if r["branch"] == "main":
            by_ep[(r["graph_id"], r["episode_id"])].append(r)

    model.eval()
    per_step_f1, count_mae, final_f1 = [], [], []

    for (gid, _eid), ers in by_ep.items():
        ers.sort(key=lambda r: r["t"])
        n = store[gid]["num_nodes"]
        adj_map = reconstruct_episode_adjacency(ers, store[gid]["base_edges"])

        # Initialise the autoregressive state from the first recorded true state
        cur_inf = set(ers[0]["state"]["infected"])
        cur_fr = set(ers[0]["state"]["frontier"])

        for r in ers:
            ei, w = edges_to_arrays(adj_map[(r["t"], "main")])

            # Replace the record's state with the rolled-out state for feature building
            rolled = dict(r)
            rolled["state"] = {
                "infected": sorted(cur_inf),
                "frontier": sorted(cur_fr),
            }
            X, _, _ = build_features(rolled, ei, n)
            gi = build_graph_input(ei, w, n, diffusion_model, device)

            prob = (
                torch.sigmoid(model(torch.from_numpy(X).to(device), gi)).cpu().numpy()
            )
            pi = prob[:, 0] > threshold  # (N,) bool
            pf = prob[:, 1] > threshold  # (N,) bool

            # Ground-truth next state from the record.
            yi = np.zeros(n, dtype=np.float32)
            yi[r["next_state"]["infected"]] = 1.0
            it = np.zeros(n, dtype=np.float32)
            it[r["state"]["infected"]] = 1.0

            # New-infection F1: evaluate only on susceptible nodes (it == 0)
            sus_mask = it == 0
            per_step_f1.append(binary_f1(pi & sus_mask, yi.astype(bool) & sus_mask))
            count_mae.append(abs(float(pi.sum()) - len(r["next_state"]["infected"])))

            # Advance the autoregressive state.
            cur_inf = set(int(v) for v in np.nonzero(pi)[0].tolist())
            cur_fr = set(int(v) for v in np.nonzero(pf)[0].tolist())

        # Final-state F1 compares rolled-out endpoint to the true endpoint
        truth_final = np.zeros(n, dtype=np.float32)
        truth_final[ers[-1]["next_state"]["infected"]] = 1.0
        rolled_final = np.zeros(n, dtype=np.float32)
        rolled_final[list(cur_inf)] = 1.0
        final_f1.append(binary_f1(rolled_final, truth_final))

    return {
        "rollout_newinf_f1": float(np.mean(per_step_f1)) if per_step_f1 else 0.0,
        "rollout_count_mae": float(np.mean(count_mae)) if count_mae else 0.0,
        "rollout_final_f1": float(np.mean(final_f1)) if final_f1 else 0.0,
    }


def rebuild_simulator(store_entry: dict, diffusion_model: str, seed: int = 0) -> object:
    """Reconstruct a data/wm_simulator.Simulator from a stored graph."""
    ei = store_entry["edge_index"]
    ic = store_entry["ic_probs"]
    n = store_entry["num_nodes"]
    directed = bool(store_entry["meta"].get("directed", False))
    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1]))
    ic_prob_map = {
        (int(ei[0, i]), int(ei[1, i])): float(ic[i]) for i in range(ei.shape[1])
    }
    sim = Simulator(g, ic_prob_map=ic_prob_map, seed=seed)
    sim.reset(diffusion_model)

    return sim


@torch.inference_mode()
def planning_regret(
    model: torch.nn.Module,
    store_entry: dict,
    diffusion_model: str,
    device: torch.device,
    n_states: int = 20,
    n_candidates: int = 20,
    mc: int = 8,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict[str, float]:
    """One-step greedy: model picks argmax predicted spread; compare true spread to oracle."""
    rng = np.random.default_rng(seed)
    n = store_entry["num_nodes"]
    ei = store_entry["edge_index"]
    ic = store_entry["ic_probs"]

    gi = build_graph_input(ei, ic, n, diffusion_model, device)
    deg = np.zeros(n)
    np.add.at(deg, ei[0], 1)
    np.add.at(deg, ei[1], 1)

    model.eval()
    reg_model, reg_rand, reg_deg = [], [], []

    def true_spread(infected: set[int], action: list) -> float:
        gains = []
        for _ in range(mc):
            sim = rebuild_simulator(
                store_entry, diffusion_model, seed=int(rng.integers(1 << 30))
            )

            for v in infected:
                sim.model.status[v] = 1

            s2 = sim.advance(action)
            gains.append(len(s2.infected) - len(infected))

        return float(np.mean(gains))

    for _ in range(n_states):
        k = max(1, n // 20)
        infected = set(rng.choice(n, size=k, replace=False).tolist())
        susceptible = [v for v in range(n) if v not in infected]

        if not susceptible:
            continue

        cands = list(
            rng.choice(
                susceptible, size=min(n_candidates, len(susceptible)), replace=False
            )
        )
        pred = []
        for v in cands:
            rec = {
                "state": {"infected": sorted(infected), "frontier": sorted(infected)},
                "action": [{"op": "add_node", "target": int(v)}],
                "next_state": {"infected": [], "frontier": []},
            }
            X, _, _ = build_features(rec, ei, n)
            prob = (
                torch.sigmoid(model(torch.from_numpy(X).to(device), gi)).cpu().numpy()
            )
            susceptible_mask = np.array([v not in infected for v in range(n)])
            pred.append(prob[susceptible_mask, 0].sum())

        a_model = cands[int(np.argmax(pred))]
        a_rand = cands[int(rng.integers(len(cands)))]
        a_deg = cands[int(np.argmax([deg[v] for v in cands]))]
        trues = {
            v: true_spread(infected, [ActionOp("add_node", int(v))]) for v in cands
        }
        oracle = max(trues.values())
        reg_model.append(oracle - trues[a_model])
        reg_rand.append(oracle - trues[a_rand])
        reg_deg.append(oracle - trues[a_deg])

    return {
        "plan_regret_model": float(np.mean(reg_model)) if reg_model else 0.0,
        "plan_regret_random": float(np.mean(reg_rand)) if reg_rand else 0.0,
        "plan_regret_degree": float(np.mean(reg_deg)) if reg_deg else 0.0,
    }


@torch.inference_mode()
def planning_regret_multi(
    model: torch.nn.Module,
    store: dict[str, dict],
    diffusion_model: str,
    device: torch.device,
    n_graphs: int = 5,
    n_states: int = 20,
    n_candidates: int = 20,
    mc: int = 8,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Average planning regret over the first n_graphs graphs in the store.

    Single-graph planning regret ties across backbones because the per-graph
    argmax choice is coarse (most models pick the same candidate). Averaging over
    several graphs — each with its own seed offset so the sampled states differ —
    gives the metric real resolution, plus a cross-graph std as an error bar.
    """
    gids = list(store)[:n_graphs]
    if not gids:
        raise ValueError("planning_regret_multi: empty graph store")

    per_graph = [
        planning_regret(
            model,
            store[gid],
            diffusion_model,
            device,
            n_states=n_states,
            n_candidates=n_candidates,
            mc=mc,
            seed=seed + i,
            threshold=threshold,
        )
        for i, gid in enumerate(gids)
    ]

    out = {"plan_n_graphs": float(len(per_graph))}
    for key in ("plan_regret_model", "plan_regret_random", "plan_regret_degree"):
        vals = np.array([pg[key] for pg in per_graph], dtype=np.float64)
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())

    return out
