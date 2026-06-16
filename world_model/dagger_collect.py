"""
Batch DAgger collection (Lever 2): roll the trained model short and free-running
on the TRAIN episodes, collect the MILD-drift states it visits, relabel them with
the simulator (correct MC marginals), and APPEND them as train-only transitions so
the next round's training sees correct (bounded) targets on its own visited states.

Design choices (see spec):
- Roll under NULL action on the BASE graph. The saturation is a diffusion phenomenon,
  and this sidesteps edge-action graph reconstruction. Drifted states are relabeled
  on the same base graph, so targets are exact.
- Collect depths 1..--depth from each episode's post-seed state: mild drift that
  brackets the true cascade's stopping point, where "no more new infections" is the
  correcting signal. (Full-saturation states relabel uselessly to "stay saturated".)
- Each round's records get a distinct episode_id (|dg<round>) so they form their own
  group -> base-graph adjacency, and a distinct branch (dagger_<round>).

Workflow (per round):
    cp -r data/output/ba20_marg_perturb data/output/ba20_dagger      # once
    python world_model/dagger_collect.py \
        --results world_model/checkpoints/ba20_marg_perturb_sage_IC.json \
        --out-dir data/output/ba20_dagger --round 1 --depth 4
    sbatch train.sbatch IC data/output/ba20_dagger
    python world_model/eval_rollout_ensemble.py --results .../ba20_dagger_sage_IC.json ...
    # repeat with --results .../ba20_dagger_sage_IC.json --round 2, etc.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from wm_data import (
    load_graph_store,
    build_features,
    build_graph_input,
    edges_to_arrays,
)
from wm_eval import rebuild_simulator
from eval_planning import load_trained_model


def make_dagger_record(
    gid: str,
    diffusion_model: str,
    episode_id: str,
    round_id: int,
    t: int,
    infected: set[int],
    frontier: set[int],
    next_state,
    inf_marg: dict[int, float],
    fr_marg: dict[int, float],
) -> dict:
    inf = sorted(int(v) for v in infected)
    fr = sorted(int(v) for v in frontier)
    return {
        "graph_id": gid,
        "diffusion_model": diffusion_model,
        "episode_id": episode_id,
        "algorithm": "dagger",
        "branch": f"dagger_{round_id}",
        "t": int(t),
        "state": {
            "infected": inf,
            "frontier": fr,
            "infected_count": len(inf),
            "frontier_count": len(fr),
        },
        "action": [],
        "next_state": next_state.to_dict(),
        "reward": float(len(next_state.infected) - len(inf)),
        "next_marginal_infected": {str(k): round(v, 6) for k, v in inf_marg.items()},
        "next_marginal_frontier": {str(k): round(v, 6) for k, v in fr_marg.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch DAgger: collect + relabel the model's mild-drift states, append to train"
    )
    parser.add_argument("--results", required=True, help="results JSON of the model to roll")
    parser.add_argument("--out-dir", required=True, help="dataset dir to append dagger transitions to")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--depth", type=int, default=4, help="rollout depth = max drift steps collected")
    parser.add_argument("--mc", type=int, default=30, help="MC draws for relabeling")
    parser.add_argument("--max-episodes", type=int, default=0, help="cap episodes rolled (0 = all)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    config = json.loads(Path(args.results).read_text())["config"]
    diffusion_model = config["diffusion_model"]
    data_dir = config["data_dir"]

    model = load_trained_model(config, device)
    model.eval()
    store = load_graph_store(data_dir)

    # Roll on the base (main-branch) train episodes.
    train_path = Path(data_dir) / f"transitions_{diffusion_model}_train.jsonl"
    by_ep = defaultdict(list)
    for line in train_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["branch"] == "main":
            by_ep[(r["graph_id"], r["episode_id"])].append(r)

    ep_keys = list(by_ep)
    if args.max_episodes and len(ep_keys) > args.max_episodes:
        ep_keys = [ep_keys[i] for i in rng.choice(len(ep_keys), size=args.max_episodes, replace=False)]

    out_records = []
    drift_sizes = []

    with torch.inference_mode():
        for gid, eid in ep_keys:
            ers = sorted(by_ep[(gid, eid)], key=lambda r: r["t"])
            n = store[gid]["num_nodes"]
            ei, w = edges_to_arrays(store[gid]["base_edges"])  # NULL action -> base graph
            gi = build_graph_input(ei, w, n, diffusion_model, device)

            seed_state = ers[0]["next_state"]  # post-seed (after t=0 seeds + first diffusion)
            cur_inf = set(seed_state["infected"])
            cur_fr = set(seed_state["frontier"])

            for d in range(args.depth + 1):
                if d >= 1:
                    # Relabel the drifted state under NULL action via the true simulator.
                    sim = rebuild_simulator(
                        store[gid], diffusion_model, seed=int(rng.integers(1 << 30))
                    )
                    sim.set_state(cur_inf, cur_fr)
                    s_next, inf_marg, fr_marg = sim.advance_marginal([], args.mc)
                    out_records.append(
                        make_dagger_record(
                            gid, diffusion_model, f"{eid}|dg{args.round}", args.round,
                            d, cur_inf, cur_fr, s_next, inf_marg, fr_marg,
                        )
                    )
                    drift_sizes.append(len(cur_inf))

                if d < args.depth:
                    # Advance the model one NULL-diffusion step (thresholded readout).
                    rolled = {
                        "state": {"infected": sorted(cur_inf), "frontier": sorted(cur_fr)},
                        "action": [],
                        "next_state": {"infected": [], "frontier": []},
                    }
                    X, _, _ = build_features(rolled, ei, n)
                    prob = (
                        torch.sigmoid(model(torch.from_numpy(X).to(device), gi)).cpu().numpy()
                    )
                    cur_inf = set(np.nonzero(prob[:, 0] > args.threshold)[0].tolist())
                    cur_fr = set(np.nonzero(prob[:, 1] > args.threshold)[0].tolist())

    out_path = Path(args.out_dir) / f"transitions_{diffusion_model}_train.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    mean_drift = float(np.mean(drift_sizes)) if drift_sizes else 0.0
    print(
        f"[dagger r{args.round}] appended {len(out_records)} transitions "
        f"({len(ep_keys)} episodes x depth {args.depth}) -> {out_path}"
    )
    print(f"[dagger r{args.round}] mean drifted infected-count = {mean_drift:.1f} "
          f"(want mild: brackets the true cascade size, not full saturation)")
