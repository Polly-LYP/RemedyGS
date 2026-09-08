import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import DiscriminatorNet

IMG_EXTS = {".jpg", ".jpeg", ".png"}
# Training resolution of the detector (paper Sec. 4.1: images resized to 960x528).
CROP_H, CROP_W = 528, 960


class SceneImages(Dataset):
    """All images under <root>/<scene>/images for every scene in dataset_list."""

    def __init__(self, root, dataset_list):
        self.samples = []  # (image_path, scene)
        with open(dataset_list) as f:
            scenes = [line.strip() for line in f if line.strip()]
        for scene in scenes:
            img_dir = Path(root) / scene / "images"
            if not img_dir.is_dir():
                print(f"[warn] skip {scene}: {img_dir} not found")
                continue
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(p), scene))
        print(f"Found {len(self.samples)} images across {len(scenes)} scenes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, scene = self.samples[index]
        image = transforms.ToTensor()(Image.open(path).convert("RGB"))
        _, h, w = image.shape
        if h >= CROP_H and w >= CROP_W:
            # Center crop keeps native-resolution textures (deterministic);
            # poisoned textures are local, so any crop carries the signature.
            image = tf.crop(image, (h - CROP_H) // 2, (w - CROP_W) // 2, CROP_H, CROP_W)
        else:
            image = tf.resize(image, [CROP_H, CROP_W])
        return image, path, scene


def run_detector(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SceneImages(args.image_dataset_path, args.dataset_list)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    net = DiscriminatorNet().to(device)
    checkpoint = torch.load(args.load_existing_model_path, map_location=device)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    net.load_state_dict(state_dict)
    net.eval()

    rows = []  # (image_path, scene, pred, prob_poisoned)
    with torch.no_grad():
        for images, paths, scenes in tqdm(dataloader):
            logits = net(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(poisoned)
            preds = logits.argmax(dim=1)
            for path, scene, pred, prob in zip(paths, scenes, preds, probs):
                rows.append((path, scene, int(pred), float(prob)))

    os.makedirs(args.output_path, exist_ok=True)
    pred_file = os.path.join(args.output_path, "detector_predictions.txt")
    with open(pred_file, "w") as f:
        f.write("image_path\tscene\tprediction(1=poisoned,0=clean)\tprob_poisoned\n")
        for path, scene, pred, prob in rows:
            f.write(f"{path}\t{scene}\t{pred}\t{prob:.4f}\n")

    poisoned = [r[0] for r in rows if r[2] == 1]
    with open(os.path.join(args.output_path, "poisoned_images.txt"), "w") as f:
        f.write("\n".join(poisoned) + ("\n" if poisoned else ""))

    per_scene = defaultdict(lambda: [0, 0])  # scene -> [poisoned, total]
    for _, scene, pred, _ in rows:
        per_scene[scene][0] += pred
        per_scene[scene][1] += 1

    print("\n===== Detector summary =====")
    for scene, (n_poi, n_all) in sorted(per_scene.items()):
        verdict = "POISONED" if n_poi * 2 > n_all else "clean"
        print(f"{scene}: {n_poi}/{n_all} images poisoned -> scene {verdict}")
    print(f"Total: {len(poisoned)}/{len(rows)} images flagged as poisoned.")
    print(f"Per-image predictions written to {pred_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RemedyGS detector inference")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_dataset_path", type=str, default="./example/MIP_Nerf_360_eps16")
    parser.add_argument("--dataset_list", type=str, default="./example/dataset_list.txt")
    parser.add_argument("--output_path", type=str, default="./example/output")
    parser.add_argument("--load_existing_model_path", type=str, default="./ckpt/D_ckpt.pth")
    args = parser.parse_args()
    run_detector(args)
