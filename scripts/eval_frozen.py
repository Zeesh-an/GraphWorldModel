"""
Evaluate a FROZEN checkpoint on a dataset it was not trained on.

    python -m scripts.eval_frozen \
        --checkpoint results/influence_maximization/ba100/wm_structured_IC/wm_sage_IC.pt \
        --data-dir results/ood/ba500/data \
        --out results/ood/ba500/eval_structured_IC.json

This is the Stage-B workhorse and it deliberately cannot train. `train_wm.py`
evaluates as a side effect of training, which makes "evaluate this model over
there" awkward enough that people retrain instead — and a target-trained model
answers a different question than a transferred one.

Two things it reports that a same-distribution eval does not need:

  * the ACTION metrics on the target distribution. Q1's current evidence is all
    in-distribution, which only shows the model uses actions on graphs like its
    training graphs. Rerunning effect_mae_norm / pearson / shuffle_drop on an OOD
    target asks the stronger question: does the learned action MECHANISM survive
    the shift? That upgrade is why this script runs the whole suite rather than
    just prediction error.

  * provenance proving the model never saw these graphs. It refuses to run if the
    target dataset has a train split, unless --allow-trained-target is passed,
    because an OOD number measured on a dataset the model could have trained on
    is not an OOD number.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from world_model.checkpoint import describe, load_checkpoint
from world_model.wm_action_eval import action_conditioning_report
from world_model.wm_data import TransitionDataset, load_graph_store
from world_model.wm_eval import evaluate_one_step, rollout_ensemble

# Torch grabs every core by default. On the tensors this repository actually
# evaluates -- a 100-to-1000 node graph, one record at a time -- that is
# catastrophic: measured on a 100-node OOD set, 14 threads cost 80.0 ms/record
# against 2.02 ms/record at 1 thread, a 40x slowdown, because the intra-op
# synchronisation dwarfs the arithmetic. It also makes parallel eval jobs fight.
# 1 is therefore the default here, not a tuning choice.
default_threads = 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Full evaluation suite for a frozen checkpoint on any dataset"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--config", type=Path, default=None,
                        help="results JSON, only needed for a legacy checkpoint")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-action", action="store_true")
    parser.add_argument(
        "--allow-trained-target",
        action="store_true",
        help="permit a target dataset that has a train split. Off by default: an "
             "OOD number measured where the model could have trained is not one.",
    )
    parser.add_argument("--threads", type=int, default=default_threads,
                        help=f"torch intra-op threads (default {default_threads}; "
                             f"more is much slower on small graphs)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    model, spec, train_meta = load_checkpoint(
        args.checkpoint, config=args.config, device=device, strict_spec=False
    )
    diffusion_model = spec.diffusion_model

    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "metadata.json"
    target_metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    target_split_mode = target_metadata.get("split_mode", "unknown")
    source_data_dir = train_meta.get("data_dir")
    same_dataset = source_data_dir and Path(source_data_dir).resolve() == data_dir.resolve()

    if (
        target_split_mode != "eval_only"
        and not same_dataset
        and not args.allow_trained_target
        and (data_dir / f"transitions_{diffusion_model}_train.jsonl").exists()
    ):
        raise ValueError(
            f"{data_dir} has a train split (split_mode={target_split_mode!r}), so a "
            f"number measured here is not evidence of transfer unless you know "
            f"this model did not train on it. Regenerate the target with "
            f"--split-mode eval_only, or pass --allow-trained-target deliberately."
        )

    print(f"[frozen] {describe(args.checkpoint)}")
    print(f"[frozen] target {data_dir} split={args.split} "
          f"split_mode={target_split_mode}")

    started = time.perf_counter()
    dataset = TransitionDataset(
        data_dir, diffusion_model, args.split, spec.action_encoding
    )
    store = load_graph_store(data_dir)

    results = {
        "question": "Q3_generalization + Q1_action_under_shift",
        "frozen": True,
        "checkpoint": str(args.checkpoint),
        "model": {
            "backbone": spec.backbone, "head": spec.head,
            "diffusion_model": diffusion_model,
            "remove_semantics": spec.remove_semantics,
            "action_encoding": spec.action_encoding,
            "hide_edge_weights": spec.hide_edge_weights,
        },
        "source": {
            "data_dir": source_data_dir,
            "split_mode": train_meta.get("split_mode"),
            "seed": train_meta.get("seed"),
        },
        "target": {
            "data_dir": str(data_dir),
            "split": args.split,
            "split_mode": target_split_mode,
            "dataset": target_metadata.get("config", {}).get("dataset"),
            "syn_nodes": target_metadata.get("config", {}).get("syn_nodes"),
            "n_graphs": len(store),
            "n_transitions": len(dataset),
            "is_same_dataset_as_source": bool(same_dataset),
            "target_has_train_split": (
                data_dir / f"transitions_{diffusion_model}_train.jsonl"
            ).exists(),
        },
        "test": evaluate_one_step(
            model, dataset, diffusion_model, device,
            hide_edge_weights=spec.hide_edge_weights,
        ),
    }

    if not args.skip_rollout:
        results["rollout"] = rollout_ensemble(
            model, str(data_dir), diffusion_model, store, device,
            split=args.split, n_samples=args.n_samples,
            max_episodes=args.max_episodes, seed=args.seed,
            remove_semantics=spec.remove_semantics,
            hide_edge_weights=spec.hide_edge_weights,
            action_encoding=spec.action_encoding,
        )

    if not args.skip_action:
        # The Q1-under-shift upgrade. If the split carries no counterfactual
        # pairs the report says UNTESTABLE rather than passing by default.
        results["action_conditioning"] = action_conditioning_report(
            model, dataset, diffusion_model, device,
            hide_edge_weights=spec.hide_edge_weights, seed=args.seed,
        )

    results["runtime_seconds"] = round(time.perf_counter() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))

    test = results["test"]
    print(f"  delta_f1              {test['delta_f1']:.4f}")
    print(f"  brier_infected        {test['brier_infected']:.5f}")
    print(f"  add_seed_success      {test['add_seed_success']:.4f}")
    print(f"  remove_frontier_succ  {test['remove_frontier_success']:.4f}")

    if "rollout" in results:
        rollout = results["rollout"]
        print(f"  ens_marg_mae          {rollout['ens_marg_mae']:.4f}")
        print(f"  ens_count_bias        {rollout['ens_count_bias']:+.3f}")
        print(f"  final model / true    {rollout['ens_final_count_model']:.2f} / "
              f"{rollout['ens_final_count_true']:.2f}")

    if "action_conditioning" in results:
        effect = results["action_conditioning"]["counterfactual_effect"]
        ablation = results["action_conditioning"]["ablation"]
        print(f"  effect_mae_norm       {effect['effect_mae_norm']:.4f}  (null 1.0)")
        print(f"  effect_pearson        {effect['effect_pearson']:.4f}")
        print(f"  shuffle_delta_f1_drop {ablation.get('shuffle_delta_f1_drop', float('nan')):.4f}")
        print(f"  verdict               {results['action_conditioning']['verdict'][:70]}")

    print(f"\n-> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
