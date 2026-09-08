<div align="center">

# RemedyGS: Defend 3D Gaussian Splatting Against Computation Cost Attacks

Official PyTorch implementation of our paper:

**RemedyGS: Defend 3D Gaussian Splatting Against Computation Cost Attacks**

[Yanping Li](https://github.com/Polly-LYP) · [Zhening Liu](https://www.liuzhening.top) · [Zijian Li](https://zli999.github.io/zijianli.github.io/) · [Zehong Lin](https://zhlinup.github.io/) · [Jun Zhang](https://eejzhang.people.ust.hk/)

[[Paper](https://arxiv.org/abs/2511.22147)] · [[Code](https://github.com/Polly-LYP/RemedyGS)] · [[Dataset](https://huggingface.co/datasets/ZincL/RemedyGS)] · [[Detector Checkpoint](https://huggingface.co/ZincL/RemedyGS-detector)]

</div>

## :star: Overview

3D Gaussian Splatting (3DGS) has become a mainstream technique for photorealistic 3D reconstruction and is widely deployed in real-world services. Recent work such as [Poison-splat](https://arxiv.org/abs/2410.08190) shows that attackers can poison input images to trigger excessive Gaussian densification, dramatically increasing GPU memory, training time, and rendering latency—potentially leading to denial-of-service (DoS) in 3DGS-as-a-service systems.

**RemedyGS** is the first effective **black-box defense framework** against such computation cost attacks. Our pipeline consists of two key components:

1. **Detector** — identifies poisoned input images with abnormal high-frequency textures induced by the attack.
2. **Purifier** — recovers benign images from poisoned counterparts via an encoder–decoder with skip connections.

We further incorporate **adversarial training** with a discriminator to align the distribution of purified images with natural clean images, improving perceptual quality and reconstruction fidelity. Extensive experiments on NeRF-Synthetic, Mip-NeRF 360, and Tanks-and-Temples demonstrate that RemedyGS achieves state-of-the-art performance in both **safety** (Gaussian count, GPU memory) and **utility** (PSNR, SSIM, LPIPS) against white-box, black-box, and adaptive attacks.

**Pipeline:** poisoned images → **Detector** (poisoned vs. clean) → **Purifier** (encoder–decoder + skip connections) → defended images → vanilla 3DGS training. Adversarial training with a **Discriminator** aligns purified outputs with the clean image distribution. (See Figure 1 in the paper.)

## :sparkles: Performance Highlights

Compared with baselines (image smoothing and limiting Gaussian number), RemedyGS:

- Reduces poisoned Gaussian counts from **~7M → ~2.5M** on Mip-NeRF 360 (avg.), close to clean (~3.2M).
- Cuts peak GPU memory on attacked scenes from **~24 GB → ~10 GB** (MIP avg.).
- Improves PSNR by up to **~4 dB** and SSIM by **~0.24** over naive defenses.
- Preserves utility on **clean** inputs: the detector bypasses purification for benign images, keeping reconstruction quality nearly identical to vanilla 3DGS.

For full quantitative results, please refer to our paper (Tables 1–5).

## ✨ Quick Start

After cloning the repository, follow these steps to set up the environment and run defense inference plus victim evaluation.

### 1. Clone the repository

```bash
git clone https://github.com/Polly-LYP/RemedyGS.git
cd RemedyGS
git submodule update --init --recursive
```

### 2. Create a conda environment

We recommend **Python 3.10+** on **Linux with an NVIDIA GPU (CUDA)**.

```bash
conda create -n remedygs python=3.10 -y
conda activate remedygs
```

Install PyTorch with CUDA support (choose the command matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
# Example: CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Build CUDA extensions (required for victim evaluation)

These packages are **not on PyPI** and must be compiled from the git submodules in this repo. See [LOCAL_PACKAGES.md](LOCAL_PACKAGES.md) for details.

```bash
cd victim/gaussian-splatting
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
cd ../..
```

> **Note:** If `setup.py` or `.cu` files under `submodules/` are empty, run `git submodule update --init --recursive` again to fetch the full upstream sources before building.

### 5. Prepare data and checkpoint

**Defense checkpoints**:

```
ckpt/D_ckpt.pth       # Detector (poisoned vs. clean image classifier) — download from HuggingFace
ckpt/G_ckpt.pth.tar   # Purifier (encoder-decoder generator) — included in repo
```

The purifier checkpoint is included in this repo. Download the detector checkpoint (~267 MB) from [ZincL/RemedyGS-detector](https://huggingface.co/ZincL/RemedyGS-detector):

```bash
pip install -U huggingface_hub
hf download ZincL/RemedyGS-detector D_ckpt.pth --repo-type model --local-dir ckpt/
# or: wget https://huggingface.co/ZincL/RemedyGS-detector/resolve/main/D_ckpt.pth -O ckpt/D_ckpt.pth
```

**Poisoned evaluation scenes** (COLMAP format: `images/` + `sparse/0/`):

```
example/MIP_Nerf_360_eps16/
  bonsai/
    images/
    sparse/
  counter/
    images/
    sparse/
  kitchen/
    images/
    sparse/
  ...
```

Scene names used during inference are listed in `example/dataset_list.txt`. Poisoned Mip-NeRF 360 scenes can be generated with the official [Poison-splat](https://github.com/jiahaolu97/poison-splat) codebase (e.g., `eps=16`).

### 6. (Optional) Download the RemedyGSData benchmark

We release **RemedyGSData**, the large-scale poisoned 3DGS training dataset used in our paper, on HuggingFace: [ZincL/RemedyGS](https://huggingface.co/datasets/ZincL/RemedyGS).

- **702 real-world scenes** derived from DL3DV-10K, poisoned by the bounded Poison-splat attacker at 7 perturbation budgets (`eps ∈ {16, 26, 35, 40, 50, 100, 150}` ×1/255), ~975 GB in total.
- Standard COLMAP format (`images/` + `sparse/0/`), drop-in compatible with vanilla 3DGS training; per-scene `.tar` shards with `scene_metadata.csv`.

```bash
pip install -U huggingface_hub

# everything (~975 GB)
hf download ZincL/RemedyGS --repo-type dataset

# or a single scene
hf download ZincL/RemedyGS eps16~eps150/eps16_<scene_hash>.tar --repo-type dataset
```

## :rocket: Run

### Detector inference

Classify input images as poisoned or clean with the trained detector (four stacked 2D conv layers + linear head, binary output: 1 = poisoned, 0 = clean):

```bash
bash scripts/detector.sh
```

This runs `inference/detector_inference.py` with:

| Argument | Default |
|----------|---------|
| Input scenes | `./example/MIP_Nerf_360_eps16` |
| Scene list | `./example/dataset_list.txt` |
| Checkpoint | `./ckpt/D_ckpt.pth` (download from [HuggingFace](https://huggingface.co/ZincL/RemedyGS-detector), see above) |
| Output | `./example/output/detector_predictions.txt` (per-image) + `poisoned_images.txt` (flagged list) |

Only images flagged as poisoned need to be passed to the purifier below; clean images are kept as-is to preserve utility.

### Defense inference

Purify poisoned images with the trained purifier and write defended scenes to disk:

```bash
bash scripts/inference.sh
```

This runs `inference/inference.py` with:

| Argument | Default |
|----------|---------|
| Input scenes | `./example/MIP_Nerf_360_eps16` |
| Scene list | `./example/dataset_list.txt` |
| Checkpoint | `./ckpt/G_ckpt.pth.tar` |
| Output | `./example/output/defended_figs/<dataset>/<scene>/images/` (+ copied `sparse/`) |

You can also override arguments directly:

```bash
export PYTHONPATH="${PWD}/inference:${PYTHONPATH:-}"
python inference/inference.py \
  --image_dataset_path ./example/MIP_Nerf_360_eps16 \
  --dataset_list ./example/dataset_list.txt \
  --output_path ./example/output \
  --load_existing_model_path ./ckpt/G_ckpt.pth.tar
```

### Victim 3DGS benchmark

Train a victim 3DGS model on (poisoned or defended) scenes and report safety metrics (Gaussian count, GPU memory, training time) and utility metrics (PSNR, SSIM, LPIPS):

```bash
bash scripts/victim.sh
```

By default, `scripts/victim.sh` runs **bonsai** and **kitchen** from `./example/MIP_Nerf_360_eps16/` in parallel. Edit the script to change scene names or GPU ids (`--gpu`).

To evaluate on **defended** images after running inference:

```bash
# Example: single scene on GPU 0
python ./victim/gaussian-splatting/benchmark.py --gpu 0 \
  -s ./example/output/defended_figs/MIP_Nerf_360_eps16/kitchen/ \
  -m ./output/victim/kitchen_defended/
```

To compare renders against **clean** ground-truth images (for PSNR/LPIPS/SSIM with raw metrics):

```bash
CLEAN_DATASET_ROOT=/path/to/clean/MIP_Nerf_360 \
python ./victim/gaussian-splatting/benchmark.py --gpu 0 \
  -s ./example/output/defended_figs/MIP_Nerf_360_eps16/kitchen/ \
  -m ./output/victim/kitchen_defended/
```

> **Note:** Each benchmark run performs 3 independent training runs (`--exp_runs 3`) with 30,000 iterations by default. Expect several hours of GPU time per scene.

## 📁 Project layout

```
RemedyGS/
├── ckpt/                          # Defense checkpoints (D_ckpt.pth detector, G_ckpt.pth.tar purifier)
├── inference/                     # Detector & purifier inference code
├── victim/gaussian-splatting/     # 3DGS victim training (benchmark)
├── scripts/
│   ├── detector.sh                # Detector inference (poisoned vs. clean)
│   ├── inference.sh               # Purifier inference
│   └── victim.sh                  # Victim benchmark (parallel scenes)
├── example/
│   ├── MIP_Nerf_360_eps16/        # Poisoned scenes (user-provided / local)
│   ├── dataset_list.txt           # Scene names for inference
│   └── output/                    # Defended images (generated)
├── requirements.txt               # Pip-installable dependencies
└── LOCAL_PACKAGES.md              # CUDA extensions install guide
```

## 🤝 Acknowledgments

We thank the authors of the following open-source projects:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Poison-splat](https://github.com/jiahaolu97/poison-splat)
- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
- [DL3DV-10K](https://github.com/DL3DV-10K/DL3DV-10K)

## ✏️ Citation

If you find RemedyGS useful, please cite our paper:

```
@article{li2025remedygs,
  title={RemedyGS: Defend 3D Gaussian Splatting Against Computation Cost Attacks},
  author={Li, Yanping and Liu, Zhening and Li, Zijian and Lin, Zehong and Zhang, Jun},
  journal={arXiv preprint arXiv:2511.22147},
  year={2025}
}
```
