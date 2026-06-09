"""
Validation harness for generated world-model transition data.

Reproduces gate checks on a produced dataset directory:
    - Reward distribution spread (not collapsed),
    - Main-branch monotone active-count growth,
    - Action sensitivity (same state_t, different action -> different next_state).

python Data/action_conditioned_wm/validate_wm_data.py --dir <output_dir>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# sys.path bootstrap so flat imports and sibling reuse resolve when run as a script
_PKG_DIR = Path(__file__).resolve().parent
_DATA_DIR = _PKG_DIR.parent
for _p in (str(_PKG_DIR), str(_DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_records(out_dir: Path) -> list:
    files = sorted(Path(out_dir).glob("transitions_*.jsonl"))

    if not files:
        raise ValueError(f"no transitions_*.jsonl files found under {out_dir}")

    records = []
    for fp in files:
        for line in fp.read_text().strip().splitlines():
            if line:
                records.append(json.loads(line))

    return records


def _action_key(action: list) -> tuple:
    return tuple(sorted((a["op"], a["target"]) for a in action))


def compute_checks(out_dir: Path) -> dict:
    records = _load_records(out_dir)
    rewards = np.array([record["reward"] for record in records], dtype=float)

    # main-branch monotonicity: next infected_count >= infected_count
    main = [record for record in records if record["branch"] == "main"]
    monotone = sum(
        1
        for r in main
        if r["next_state"]["infected_count"] >= r["state"]["infected_count"]
    )
    main_monotone_ratio = monotone / len(main) if main else 0.0

    # action sensitivity: group by (graph, episode, t, state.infected), count
    # groups where differing actions produce differing next_states.
    groups = defaultdict(list)
    for record in records:
        key = (
            record["graph_id"],
            record["episode_id"],
            record["t"],
            tuple(record["state"]["infected"]),
        )
        groups[key].append(record)

    sensitivity_pairs = 0

    for recs in groups.values():
        if len(recs) < 2:
            continue
        by_action: dict[tuple[tuple[str, int], ...], set[tuple[int, ...]]] = {}
        for r in recs:
            by_action.setdefault(_action_key(r["action"]), set()).add(
                tuple(r["next_state"]["infected"])
            )
        actions = list(by_action.keys())
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                if by_action[actions[i]] != by_action[actions[j]]:
                    sensitivity_pairs += 1

    # per-algorithm mean final spread (terminal infected_count, main branch) —
    # reported (not strictly asserted) so the ranking random < degree/pagerank
    # < celf/local_search can be eyeballed without MC-flaky test failures.
    ep_final = {}
    ep_algo = {}

    for r in main:
        ek = (r["graph_id"], r["episode_id"])
        ep_algo[ek] = r["algorithm"]
        ep_final[ek] = max(ep_final.get(ek, 0), r["next_state"]["infected_count"])

    algo_spreads = defaultdict(list)

    for ek, fin in ep_final.items():
        algo_spreads[ep_algo[ek]].append(fin)

    per_algorithm_final_spread = {
        a: float(np.mean(v)) for a, v in sorted(algo_spreads.items())
    }

    return {
        "n_transitions": len(records),
        "n_main": len(main),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "main_monotone_ratio": float(main_monotone_ratio),
        "action_sensitivity_pairs": int(sensitivity_pairs),
        "per_algorithm_final_spread": per_algorithm_final_spread,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate generated WM transition data"
    )
    parser.add_argument(
        "--dir", required=True, help="output dir produced by generate_wm_data"
    )
    args = parser.parse_args()

    checks = compute_checks(Path(args.dir))
    print(json.dumps(checks, indent=2))
