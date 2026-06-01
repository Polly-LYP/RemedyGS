#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/inference:${PYTHONPATH:-}"

python inference/inference.py \
  --image_dataset_path ./example/MIP_Nerf_360_eps16 \
  --dataset_list ./example/dataset_list.txt \
  --output_path ./example/output \
  --exp_name inference_defense \
  --load_existing_model_path ./ckpt/G_ckpt.pth.tar
