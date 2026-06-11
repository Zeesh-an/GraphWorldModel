#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODEL="${1:-gcn}"; DM="${2:-IC}"
G=(--dataset ba --num-graphs 1 --syn-nodes 100 --ba-m 3 --models "$DM" \
   --algorithms random degree pagerank celf --rollouts 10 --horizon 10 --seed 42)

python data/generate_wm_data.py "${G[@]}" --out-dir data/output/abl_diffusion
python data/generate_wm_data.py "${G[@]}" --action-ops add_node remove_node --out-dir data/output/abl_node
python data/generate_wm_data.py "${G[@]}" --action-ops add_edge remove_edge set_edge_weight --out-dir data/output/abl_edge

for S in diffusion node edge; do
  python world_model/train_wm.py --data-dir "data/output/abl_${S}" --diffusion-model "$DM" \
    --model "$MODEL" --epochs 200 --plan-demo \
    --results "world_model/checkpoints/abl_${S}_${MODEL}_${DM}.json"
done
python world_model/compare_runs.py \
  world_model/checkpoints/abl_diffusion_${MODEL}_${DM}.json \
  world_model/checkpoints/abl_node_${MODEL}_${DM}.json \
  world_model/checkpoints/abl_edge_${MODEL}_${DM}.json
