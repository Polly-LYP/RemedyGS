import argparse
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataset_tool.MIPNerf_ImagesDataloader_May_test import MIPNerf_MV_Images
from model import SharpenAddNet_deeper_cGAN_v1


def my_test(net, test_dataloader, args):
    net.eval()
    test_dataloader = tqdm(test_dataloader)

    for _, (attacked_image, image_paths, without_resize_images) in enumerate(test_dataloader):
        attacked_image = attacked_image.cuda()
        if attacked_image.dim() == 5:
            attacked_image = attacked_image.squeeze(1)
        elif attacked_image.dim() == 3:
            attacked_image = attacked_image.unsqueeze(0)

        origin_h = without_resize_images.shape[-2]
        origin_w = without_resize_images.shape[-1]
        if without_resize_images.dim() == 5:
            origin_h = without_resize_images.squeeze(1).shape[-2]
            origin_w = without_resize_images.squeeze(1).shape[-1]

        _, defended = net(attacked_image)
        resize = transforms.Resize((origin_h, origin_w))

        for i in range(defended.shape[0]):
            if isinstance(image_paths[i], (list, tuple)):
                img_pth = image_paths[i][0] if len(image_paths[i]) == 1 else image_paths[i][i]
            else:
                img_pth = image_paths[i]
            parts = Path(img_pth).parts
            img_name = parts[-1]
            scene_name = parts[-3]
            attacked_name = parts[-4]

            out_dir = os.path.join(
                args.output_path, "defended_figs", attacked_name, scene_name, "images"
            )
            os.makedirs(out_dir, exist_ok=True)
            save_image(resize(defended[i].unsqueeze(0)), os.path.join(out_dir, img_name))

            source_sparse = os.path.join(os.path.dirname(os.path.dirname(img_pth)), "sparse")
            dst_sparse = os.path.join(
                args.output_path, "defended_figs", attacked_name, scene_name, "sparse"
            )
            if os.path.isdir(source_sparse) and not os.path.exists(dst_sparse):
                shutil.copytree(source_sparse, dst_sparse)


def my_Test_PoisonGS(args):
    test_dataset = MIPNerf_MV_Images(
        img_per_batch=args.img_per_batch,
        root=args.image_dataset_path,
        dataset_list=args.dataset_list,
        subset="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    with torch.no_grad():
        net = SharpenAddNet_deeper_cGAN_v1().cuda()
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

        if args.load_existing_model_path:
            assert os.path.exists(args.load_existing_model_path), (
                f"Model path {args.load_existing_model_path} does not exist."
            )
            checkpoint = torch.load(args.load_existing_model_path, map_location="cuda")
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            net.load_state_dict(state_dict)

        my_test(net, test_dataloader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RemedyGS defense inference")
    parser.add_argument("--img_per_batch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--image_dataset_path",
        type=str,
        default="./example/MIP_Nerf_360_eps16",
    )
    parser.add_argument(
        "--dataset_list",
        type=str,
        default="./example/dataset_list.txt",
    )
    parser.add_argument("--output_path", type=str, default="./example/output")
    parser.add_argument("--exp_name", type=str, default="inference_defense")
    parser.add_argument(
        "--load_existing_model_path",
        type=str,
        default="./example/model.pth",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    my_Test_PoisonGS(args)
