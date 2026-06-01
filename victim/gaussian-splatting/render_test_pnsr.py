#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render,render_visualize
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import gc
import torch
import numpy as np
import os
import sys
import random
from random import randint
import uuid
import time
import re
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state,PILtoTorch
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
import multiprocessing
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from lpipsPyTorch.modules.lpips import LPIPS
import numpy as np


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    PSNR_views_with_raw = []
    LPIPS_views_with_raw = []
    ssim_views_with_raw = []
    lpips_criterion = LPIPS('vgg', '0.1').to("cuda").eval()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


        scene_name = args.source_path.split('/')[-2]
        raw_img_pth = os.path.join("/home/ylitx/Project/25_4/RemedyGS_Apr/Main/poison-splat/dataset/Nerf_Synthetic",scene_name,'test',(view.image_name + '.png'))
        raw_image = Image.open(raw_img_pth)
        raw_im_data = np.array(raw_image.convert("RGBA"))
        raw_white_background = False
        raw_bg = np.array([1,1,1]) if raw_white_background else np.array([0, 0, 0])
        raw_norm_data = raw_im_data / 255.0
        raw_arr = raw_norm_data[:,:,:3] * raw_norm_data[:, :, 3:4] + raw_bg * (1 - raw_norm_data[:, :, 3:4])
        raw_image = Image.fromarray(np.array(raw_arr*255.0, dtype=np.byte), "RGB")
        raw_resized_image_rgb = PILtoTorch(raw_image, (gt.shape[2],gt.shape[1]))
        raw_image = raw_resized_image_rgb[:3, ...]


        zero_mask = np.all(raw_arr[:, :, :3] == 0, axis=2)
        #render_arr = np.array(rendering)
        zero_mask = np.stack([zero_mask , zero_mask , zero_mask ], axis=0)
        rendering[zero_mask] = 0


        PSNR_views_with_raw.append(psnr(raw_image.to(rendering.device),rendering).mean().item())
        LPIPS_views_with_raw.append(lpips_criterion(rendering, raw_image.to(rendering.device)).item())
        ssim_views_with_raw.append(ssim(rendering,raw_image.to(rendering.device)).item())

    mean_PSNR_with_raw = round(sum(PSNR_views_with_raw) / len(PSNR_views_with_raw), 4)
    mean_LPIPS_with_raw = round(sum(LPIPS_views_with_raw) / len(LPIPS_views_with_raw), 4)
    mean_ssim_with_raw = round(sum(ssim_views_with_raw)/len(ssim_views_with_raw),4)
    with open(f"{args.model_path}/mask_metrics.txt", "w") as f:
        f.write(f"mean_PSNR_with_raw: {mean_PSNR_with_raw}\n")
        f.write(f"mean_LPIPS_with_raw: {mean_LPIPS_with_raw}\n")
        f.write(f"mean_ssim_with_raw: {mean_ssim_with_raw}\n")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(3)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)