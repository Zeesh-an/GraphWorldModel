"""
Experiment 3's table, with the comparison PAIRED.

    python -m scripts.aggregate_feedback_tiers \
        --runs experiments/feedback/sbm_trained.json

Every tier runs on the same (graph, repetition) cells, so the across-cell spread
of the trusted spread is mostly graph difficulty and swamps the tier effect. The
quantity with any resolution is the WITHIN-CELL difference against a reference
arm, and its standard error over cells. That is what this prints; the unpaired
means are kept beside it so a reader can see how much the pairing bought.

The reference is `f0` by default. `f0_matched` — f0 run for as many iterations
as it takes to spend the richest tier's world-model budget — is the arm that
decides whether a win belongs to the FEEDBACK or merely to the extra queries the
feedback paid for. Read the ladder against it, not only against `f0`.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--reference", type=str, default="f0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = json.loads(args.runs.read_text())
    rows = payload["runs"]
    tiers = list(dict.fromkeys(row["tier"] for row in rows))

    by_cell = defaultdict(dict)
    for row in rows:
        by_cell[(row["graph_id"], row["repetition"])][row["tier"]] = row

    cells = [cell for cell, block in by_cell.items() if set(tiers) <= set(block)]
    reference_rows = [by_cell[cell][args.reference] for cell in cells]

    table = {}
    for tier in tiers:
        subset = [by_cell[cell][tier] for cell in cells]
        trusted = np.array([row["trusted_final"] for row in subset])
        paired = trusted - np.array([row["trusted_final"] for row in reference_rows])

        table[tier] = {
            "n_cells": len(cells),
            "trusted_mean": float(trusted.mean()),
            "trusted_sd": float(trusted.std(ddof=1)) if trusted.size > 1 else 0.0,
            "paired_delta_vs_reference": float(paired.mean()),
            "paired_se": float(paired.std(ddof=1) / np.sqrt(paired.size))
            if paired.size > 1
            else 0.0,
            # The only verdict: does the paired gap clear two of its own SE
            "separated": bool(
                paired.size > 1
                and abs(paired.mean()) > 2.0 * paired.std(ddof=1) / np.sqrt(paired.size)
            ),
            "wins": int((paired > 0).sum()),
            "losses": int((paired < 0).sum()),
            "wm_rollouts_mean": float(np.mean([row["wm_rollouts"] for row in subset])),
            "trusted_episodes_mean": float(
                np.mean([row["trusted_episodes"] for row in subset])
            ),
            "iterations_mean": float(np.mean([row["iterations"] for row in subset])),
            "feedback_chars_mean": float(
                np.mean([row["feedback_chars"] for row in subset])
            ),
            "diagnostic_simulator_episodes": int(
                sum(row["diagnostic_simulator_episodes"] for row in subset)
            ),
        }

    references = payload.get("references", {})
    degree = [block["degree"]["trusted_final"] for block in references.values()]
    random_reference = [block["random"]["trusted_final"] for block in references.values()]

    header = (
        f"{'arm':<12}{'trusted':>10}{'vs ' + args.reference:>14}{'2SE?':>7}"
        f"{'W-L':>8}{'wm_calls':>10}{'trusted_ep':>12}{'iters':>7}{'chars':>9}"
    )
    print(f"cells (graph x repetition): {len(cells)}")
    print(header)
    for tier, block in table.items():
        print(
            f"{tier:<12}"
            f"{block['trusted_mean']:>10.2f}"
            f"{block['paired_delta_vs_reference']:>+9.2f}±{block['paired_se']:<4.2f}"
            f"{'yes' if block['separated'] else 'no':>7}"
            f"{str(block['wins']) + '-' + str(block['losses']):>8}"
            f"{block['wm_rollouts_mean']:>10.0f}"
            f"{block['trusted_episodes_mean']:>12.0f}"
            f"{block['iterations_mean']:>7.0f}"
            f"{block['feedback_chars_mean']:>9.0f}"
        )

    if degree:
        print(
            f"\nreferences on the same trusted simulator: "
            f"degree {np.mean(degree):.2f}, random {np.mean(random_reference):.2f}"
        )

    leaked = sum(block["diagnostic_simulator_episodes"] for block in table.values())
    print(
        f"trusted-simulator episodes spent INSIDE the diagnostics: {leaked} "
        f"({'clean' if leaked == 0 else 'LEAK — the tiers are not comparable'})"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "reference": args.reference,
                    "n_cells": len(cells),
                    "table": table,
                    "degree_reference": float(np.mean(degree)) if degree else None,
                    "random_reference": float(np.mean(random_reference))
                    if random_reference
                    else None,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
