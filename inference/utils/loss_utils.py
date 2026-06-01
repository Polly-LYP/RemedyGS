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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
import math

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def image_total_variation(image_tensor):
    if image_tensor.dim() == 3:
        raise ValueError("Currently we are employing batch processing. Check here.")  # For bug robustness
        return torch.sum(torch.abs(image_tensor[:, :-1, :] - image_tensor[:, 1:, :])) + \
            torch.sum(torch.abs(image_tensor[:, :, :-1] - image_tensor[:, :, 1:]))
    elif image_tensor.dim() == 4:
        return torch.sum(torch.abs(image_tensor[:, :, :, :-1] - image_tensor[:, :, :, 1:])) + \
            torch.sum(torch.abs(image_tensor[:, :, :-1, :] - image_tensor[:, :, 1:, :]))
    else:
        raise ValueError("The input tensor should be 3D or 4D tensor.")  # For bug robustness
        return None

class RenderL1_Storage_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        # self.lmbda = lmbda

    def forward(self, output_rendering, gt_image, likelihoods, N_gaussian, lmbda):
        assert gt_image.dim() == 3, "Currently we do not employ batch processing. Check here."
        # _, H, W = gt_image.size()
        # num_pixels = H * W
        out = {}
        # loss calculation
        out['bpp'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * N_gaussian))
            for likelihoods in likelihoods.values())
        out['psnr'] = psnr(output_rendering, gt_image).mean().double()
        out["l1"] = l1_loss(output_rendering, gt_image)
        out['loss'] = out['l1'] + lmbda * out['bpp']

        return out

class ReconNSharpness(nn.Module):
    """l1 Loss + sharpness loss"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        # self.lmbda = lmbda

    def forward(self, output_rendering, gt_image, lmbda):
        assert gt_image.dim() == 4, "Batch processing is empolyed"
        B, _, H, W = gt_image.shape
        num_pixels = H * W
        out = {}
        # loss calculation
        out['sharpness'] = image_total_variation(output_rendering) / (B * num_pixels)
        out["l1"] = l1_loss(output_rendering, gt_image)
        out['psnr'] = psnr(output_rendering, gt_image).mean().double()
        if out["l1"] < 0.005:
            out["loss"] = - lmbda * out["sharpness"]
        else:
            out['loss'] = out['l1'] - lmbda * out['sharpness']
        # out['loss'] = 0 - out['sharpness']   # try to optimize sharpness only
        # out['loss'] = out['l1']

        return out

class ReconNSharpnessMaxStop(nn.Module):
    """l1 Loss + sharpness loss"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        # self.lmbda = lmbda

    def forward(self, output_rendering, gt_image, lmbda):
        assert gt_image.dim() == 4, "Batch processing is empolyed"
        B, _, H, W = gt_image.shape
        num_pixels = H * W
        out = {}
        # loss calculation
        out['sharpness'] = image_total_variation(output_rendering) / (B * num_pixels)
        out["l1"] = l1_loss(output_rendering, gt_image)
        out['psnr'] = psnr(output_rendering, gt_image).mean().double()
        # out['loss'] = (torch.abs(out['l1'] - 0.02) + torch.abs(out['l1'] + 0.02)) / 2 - lmbda * out['sharpness']
        out['loss'] = torch.relu(out['l1'] - 0.01) - lmbda * out['sharpness']
        # out['loss'] = out['l1'] - lmbda * out['sharpness']

        return out

class ReconNDefense(nn.Module):
    """l1 Loss + smoothness loss"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        # self.lmbda = lmbda

    def forward(self, output_rendering, gt_image):
        assert gt_image.dim() == 4, "Batch processing is empolyed"
        B, _, H, W = gt_image.shape
        num_pixels = H * W
        out = {}
        # loss calculation
        #out['sharpness'] = image_total_variation(output_rendering) / (B * num_pixels)
        #out["l1"] = l1_loss(output_rendering, gt_image)
        #out['psnr'] = psnr(output_rendering, gt_image).mean().double()
        # out['loss'] = out['l1'] + lmbda * out['sharpness']
        out['mse'] = l2_loss(output_rendering, gt_image)
        out['ssim'] = ssim(output_rendering,gt_image)
        out['loss'] = out['mse']
        # if out["l1"] < 0.005:
        #     out["loss"] = - lmbda * out["sharpness"]
        # else:
        #     out['loss'] = out['l1'] - lmbda * out['sharpness']

        return out
