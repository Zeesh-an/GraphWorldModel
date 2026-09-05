"""
Stage B — build and validate the OOD target-distribution manifest.

    python -m scripts.build_ood_manifest \
        --source results/influence_maximization/ba100/data \
        --targets results/ood/ba200/data results/ood/er100/data ... \
        --out results/ood/manifest.json

What this proves, and why it is not a formality. "The target graphs are held
out" is the load-bearing assumption of every Stage-B number, and the usual way
it is shown — different names, different directories — proves nothing: two
generators can emit structurally identical graphs under different ids, and a
copied directory keeps its ids while sharing every edge.

So disjointness is checked STRUCTURALLY. Each graph is reduced to a canonical
hash of (num_nodes, sorted edge list), and the manifest asserts that no target
graph's hash appears anywhere in the source dataset — not just in its train
split, in ANY split, because a graph the source model saw at validation is also
not held out for our purposes.

The manifest also records measured density per family, because a topology-shift
experiment whose families differ in average degree is not isolating topology.
Stating the achieved value lets a reader check the matching rather than trust it.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from world_model.wm_data import load_graph_store


def structural_hash(entry: dict) -> str:
    """
    Canonical fingerprint of a graph's structure.

    Edges are sorted and undirected duplicates collapsed, so two graphs match
    here exactly when they are the same graph, whatever they are called and
    whichever order their edges were written in. Edge weights are excluded on
    purpose: `prob_model=weighted` derives them from the structure, so including
    them would only restate it, and a weight-only difference does not make a
    graph unseen.
    """
    edge_index = np.asarray(entry["edge_index"], dtype=np.int64)
    edges = sorted(
        {
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(edge_index[0], edge_index[1])
        }
    )
    payload = f"{int(entry['num_nodes'])}|" + ";".join(f"{a},{b}" for a, b in edges)

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe_dataset(data_dir: Path) -> dict:
    store = load_graph_store(data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    config = metadata["config"]

    graphs = {}
    node_counts, edge_counts, degrees = [], [], []

    for graph_id, entry in store.items():
        num_nodes = int(entry["num_nodes"])
        edge_index = np.asarray(entry["edge_index"], dtype=np.int64)
        undirected = {
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(edge_index[0], edge_index[1])
        }
        graphs[graph_id] = {
            "hash": structural_hash(entry),
            "num_nodes": num_nodes,
            "num_edges": len(undirected),
            "avg_degree": round(2 * len(undirected) / num_nodes, 3),
            "splits": metadata.get("splits_per_graph", {}).get(graph_id, []),
        }
        node_counts.append(num_nodes)
        edge_counts.append(len(undirected))
        degrees.append(2 * len(undirected) / num_nodes)

    return {
        "data_dir": str(data_dir),
        "dataset": config["dataset"],
        "split_mode": metadata.get("split_mode"),
        "split_is_graph_disjoint": metadata.get("split_is_graph_disjoint"),
        "n_graphs": len(graphs),
        "n_episodes": metadata.get("n_episodes"),
        # The knobs that must stay fixed for a shift experiment to isolate one
        # variable. Recorded per dataset so a mismatch is visible in the manifest
        # rather than having to be reconstructed from shell history.
        "held_fixed": {
            "models": config["models"],
            "prob_model": config["prob_model"],
            "budget_pct_range": config["budget_pct_range"],
            "algorithms": config["algorithms"],
            "rollouts": config["rollouts"],
            "horizon": config["horizon"],
            "inject_p": config["inject_p"],
            "cf_prob": config["cf_prob"],
            "cf_branches": config["cf_branches"],
            "action_ops": config["action_ops"],
            "weight_lo": config["weight_lo"],
            "weight_hi": config["weight_hi"],
            "mc_marginals": config["mc_marginals"],
            "remove_semantics": config["remove_semantics"],
            "seed": config["seed"],
        },
        "shifted": {
            "dataset": config["dataset"],
            "syn_nodes": config.get("syn_nodes"),
            "ba_m": config.get("ba_m"),
            "er_p": config.get("er_p"),
            "ws_k": config.get("ws_k"),
            "ws_p": config.get("ws_p"),
            "sbm_blocks": config.get("sbm_blocks"),
            "sbm_p_in": config.get("sbm_p_in"),
            "sbm_p_out": config.get("sbm_p_out"),
        },
        "measured": {
            "mean_num_nodes": float(np.mean(node_counts)),
            "mean_num_edges": float(np.mean(edge_counts)),
            # The number that decides whether a topology comparison is confounded.
            "mean_avg_degree": round(float(np.mean(degrees)), 3),
        },
        "graphs": graphs,
    }


def check_disjoint(source: dict, target: dict) -> dict:
    """
    Structural overlap between a target set and EVERY split of the source.

    Every split, not just train: a graph the source model early-stopped on is
    not held out either, and a target set that reuses source validation graphs
    would produce an optimistic transfer number with no train overlap at all.
    """
    source_hashes = {info["hash"]: graph_id for graph_id, info in source["graphs"].items()}
    source_train = {
        info["hash"]
        for info in source["graphs"].values()
        if "train" in info["splits"]
    }
    collisions = [
        {
            "target_graph": graph_id,
            "source_graph": source_hashes[info["hash"]],
            "hash": info["hash"],
        }
        for graph_id, info in target["graphs"].items()
        if info["hash"] in source_hashes
    ]
    train_collisions = [
        collision
        for collision in collisions
        if target["graphs"][collision["target_graph"]]["hash"] in source_train
    ]

    return {
        "source": source["data_dir"],
        "target": target["data_dir"],
        "n_source_graphs": source["n_graphs"],
        "n_target_graphs": target["n_graphs"],
        "structural_overlap_any_split": len(collisions),
        "structural_overlap_train_split": len(train_collisions),
        "collisions": collisions[:10],
        "disjoint": not collisions,
        # An eval_only target has no train split of its own, so it cannot be
        # accidentally trained on even by a script that ignores this manifest.
        "target_has_no_train_split": target["split_mode"] == "eval_only",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-B OOD manifest + disjointness proof")
    parser.add_argument("--source", type=Path, required=True,
                        help="the dataset the frozen world model was TRAINED on (default: None).")
    parser.add_argument("--targets", type=Path, nargs="+", required=True, help="OOD target dataset directories (default: None).")
    parser.add_argument("--out", type=Path, required=True, help="output path (default: None).")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any target overlaps the source (default: False).")
    args = parser.parse_args()

    source = describe_dataset(args.source)
    targets, checks = {}, []

    for target_dir in args.targets:
        target = describe_dataset(target_dir)
        targets[target["data_dir"]] = target
        checks.append(check_disjoint(source, target))

    manifest = {
        "purpose": "Stage B — graph-size and graph-topology OOD evaluation sets "
                   "for a frozen BA-100-trained IC world model",
        "training_allowed_on_targets": False,
        "source": source,
        "targets": targets,
        "disjointness": checks,
        "all_disjoint": all(check["disjoint"] for check in checks),
    }
    os.makedirs(args.out.parent, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, default=str))

    source_degree = source["measured"]["mean_avg_degree"]
    print(f"SOURCE {source['dataset']:6s} n={source['n_graphs']:3d} "
          f"N={source['measured']['mean_num_nodes']:.0f} "
          f"avg_deg={source_degree:.2f} split_mode={source['split_mode']}")
    print(f"\n{'target':28s} {'n':>3s} {'N':>6s} {'avg_deg':>8s} {'deg vs src':>11s} "
          f"{'overlap':>8s} {'eval_only':>10s}")
    print("-" * 84)

    for check in checks:
        target = targets[check["target"]]
        degree = target["measured"]["mean_avg_degree"]
        print(
            f"{Path(check['target']).parent.name:28s} "
            f"{target['n_graphs']:3d} "
            f"{target['measured']['mean_num_nodes']:6.0f} "
            f"{degree:8.2f} "
            f"{degree - source_degree:+11.2f} "
            f"{check['structural_overlap_any_split']:8d} "
            f"{str(check['target_has_no_train_split']):>10s}"
        )

    print(f"\nall targets structurally disjoint from source: {manifest['all_disjoint']}")
    print(f"-> {args.out}")

    if args.strict and not manifest["all_disjoint"]:
        raise SystemExit(1)
