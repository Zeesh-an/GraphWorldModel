"""Print a side-by-side comparison of world-model results JSONs."""

import json
import sys

KEYS = [
    "new_infection_f1",
    "delta_f1",
    "add_seed_success",
    "remove_frontier_success",
    "action_sensitivity",
]


def main(paths: list[str]) -> None:
    rows = []
    for p in paths:
        d = json.loads(open(p).read())
        t = d["test"]
        row = {"run": p.split("/")[-1].replace(".json", "")}
        for k in KEYS:
            row[k] = t.get(k)
        row["persist_delta_f1"] = t.get("persistence", {}).get("delta_f1")
        if "planning" in d:
            row["plan_regret_model"] = d["planning"]["plan_regret_model"]
            row["plan_regret_random"] = d["planning"]["plan_regret_random"]
        rows.append(row)
    cols = (
        ["run"] + KEYS + ["persist_delta_f1", "plan_regret_model", "plan_regret_random"]
    )
    print(" | ".join(cols))
    for r in rows:
        print(
            " | ".join(
                f"{r.get(c)}" if not isinstance(r.get(c), float) else f"{r[c]:.3f}"
                for c in cols
            )
        )


if __name__ == "__main__":
    main(sys.argv[1:])
