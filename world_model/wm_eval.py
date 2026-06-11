"""One-step eval, action-specific metrics, free-running rollout, planning."""

from collections import defaultdict
import numpy as np
import torch

from wm_data import (
    TransitionDataset,
    collate_transitions,
    build_features,
    build_graph_input,
    CH_INFECTED,
    CH_FRONTIER,
)
from wm_metrics import score_predictions, persistence_baseline, binary_f1


@torch.inference_mode()
def evaluate_one_step(
    model: torch.nn.Module,
    dataset: TransitionDataset,
    diffusion_model: str,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Teacher-forced one-step evaluation over an entire dataset.

    Aggregates the full metric suite (score_predictions), action-specific metrics
    (Add-Seed Success, Remove-Frontier Success), action sensitivity (how much the
    model output changes under counterfactual actions at the same state), and the
    persistence baseline.
    """
    model.eval()
    P_inf, P_fr, Y_inf, Y_fr, I_t, F_t = [], [], [], [], [], []
    # sens[state_key][action_key] = tuple of per-node predicted infected probabilities
    sens: dict[tuple, dict[tuple, tuple]] = defaultdict(dict)
    add_hits = add_tot = rem_hits = rem_tot = 0

    for i in range(len(dataset)):
        item = dataset[i]
        batch = collate_transitions([item], diffusion_model, device)
        logits = model(batch["X"], batch["graph"])  # (N, 2)
        prob = torch.sigmoid(logits).cpu().numpy()
        pi = (prob[:, 0] > threshold).astype(np.float32)
        pf = (prob[:, 1] > threshold).astype(np.float32)
        yi = item["y_inf"].numpy()
        yf = item["y_fr"].numpy()
        it = item["X"][:, CH_INFECTED].numpy()
        ft = item["X"][:, CH_FRONTIER].numpy()
        P_inf.append(pi)
        P_fr.append(pf)
        Y_inf.append(yi)
        Y_fr.append(yf)
        I_t.append(it)
        F_t.append(ft)

        r = item["record"]
        for op in r["action"]:
            if op["op"] == "add_node":
                add_tot += 1
                add_hits += int(pi[int(op["target"])] == 1)
            elif op["op"] == "remove_node":
                rem_tot += 1
                rem_hits += int(pf[int(op["target"])] == 0)

        # Group main + cf transitions by (graph, episode, t, state) for sensitivity.
        skey = (r["graph_id"], r["episode_id"], r["t"], tuple(r["state"]["infected"]))
        akey = tuple(
            sorted(
                (a["op"], a["target"], a.get("destination", -1), a.get("weight", -1.0))
                for a in r["action"]
            )
        )
        sens[skey][akey] = tuple(pi.tolist())

    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    out: dict[str, float] = score_predictions(
        cat(P_inf), cat(P_fr), cat(Y_inf), cat(Y_fr), cat(I_t), cat(F_t)
    )
    out["add_seed_success"] = add_hits / add_tot if add_tot else float("nan")
    out["remove_frontier_success"] = rem_hits / rem_tot if rem_tot else float("nan")

    # Action sensitivity: mean number of distinct output tuples per (state, multiple actions) group.
    diffs = [len(set(d.values())) - 1 for d in sens.values() if len(d) >= 2]
    out["action_sensitivity"] = float(np.mean(diffs)) if diffs else 0.0

    out["persistence"] = persistence_baseline(cat(Y_inf), cat(Y_fr), cat(I_t), cat(F_t))
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
    """Feed the model its own thresholded prediction + recorded action; compare to truth.

    For each episode (main branch only), the model is initialised with the true state
    at t=0 and then fed its own binary output at each subsequent step.  The recorded
    (ground-truth) action is applied at every step so the model sees the correct
    intervention signal.  Three metrics are reported:

    - rollout_newinf_f1   : per-step F1 on newly infected nodes only (susceptible mask)
    - rollout_count_mae   : per-step |predicted count - true count| of infected nodes
    - rollout_final_f1    : F1 between rolled-out final state and true final state
    """
    import json
    from pathlib import Path
    from wm_data import reconstruct_episode_adjacency, _edges_to_arrays

    path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    # Keep only main-branch records; group by (graph_id, episode_id).
    by_ep: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        if r["branch"] == "main":
            by_ep[(r["graph_id"], r["episode_id"])].append(r)

    model.eval()
    per_step_f1: list[float] = []
    count_mae: list[float] = []
    final_f1: list[float] = []

    for (gid, _eid), ers in by_ep.items():
        ers.sort(key=lambda r: r["t"])
        n: int = store[gid]["num_nodes"]
        adj_map = reconstruct_episode_adjacency(ers, store[gid]["base_edges"])

        # Initialise the autoregressive state from the first recorded true state.
        cur_inf: set[int] = set(ers[0]["state"]["infected"])
        cur_fr: set[int] = set(ers[0]["state"]["frontier"])

        for r in ers:
            ei, w = _edges_to_arrays(adj_map[(r["t"], "main")])

            # Replace the record's state with the rolled-out state for feature building.
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
            pi: np.ndarray = prob[:, 0] > threshold  # (N,) bool
            pf: np.ndarray = prob[:, 1] > threshold  # (N,) bool

            # Ground-truth next state from the record.
            yi = np.zeros(n, dtype=np.float32)
            yi[r["next_state"]["infected"]] = 1.0
            it = np.zeros(n, dtype=np.float32)
            it[r["state"]["infected"]] = 1.0

            # New-infection F1: evaluate only on susceptible nodes (it == 0).
            sus_mask: np.ndarray = it == 0
            per_step_f1.append(binary_f1(pi & sus_mask, yi.astype(bool) & sus_mask))
            count_mae.append(abs(float(pi.sum()) - len(r["next_state"]["infected"])))

            # Advance the autoregressive state.
            cur_inf = set(int(v) for v in np.nonzero(pi)[0].tolist())
            cur_fr = set(int(v) for v in np.nonzero(pf)[0].tolist())

        # Final-state F1 compares rolled-out endpoint to the true endpoint.
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


# Put the repo root on sys.path so the planning oracle imports as `data.wm_simulator`
# (a package path that static analysis can resolve, and that works at runtime).
import sys as _sys
from pathlib import Path as _Path

_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


def rebuild_simulator(store_entry: dict, diffusion_model: str, seed: int = 0) -> object:
    """Reconstruct a data/wm_simulator.Simulator from a stored graph."""
    import networkx as nx
    from data.wm_simulator import Simulator

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
    from data.wm_simulator import ActionOp

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
        sus = [v for v in range(n) if v not in infected]
        if not sus:
            continue
        cands = list(rng.choice(sus, size=min(n_candidates, len(sus)), replace=False))
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
            pred.append(prob[:, 0].sum())
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
