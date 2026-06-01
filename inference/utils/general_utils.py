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
import sys
from datetime import datetime
import numpy as np
import random
import time
import torchvision.transforms as transforms
import torch.optim as optim
from typing import List
import os

transform = transforms.Compose([
    transforms.ToTensor()  # 将PIL图像转换为Tensor并归一化到[0,1]之间
])

def configure_optimizers(net, learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    # for name, param in net.named_parameters():
    #     print(f"{name}: {param.dtype}")
    
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    param_groups = [
        {'params': net.ad_fe, 'lr': 0.00005},
        {'params': net.ad_op, 'lr': 0.00005},
        {'params': net.ad_sc, 'lr': 0.0005},
        {'params': net.ad_ro, 'lr': 0.0000005},
        {'params': [params_dict[n] for n in sorted(parameters) if n not in ['ad_fe', 'ad_op', 'ad_sc', 'ad_ro']], 'lr': learning_rate}
    ]
    
    optimizer = optim.Adam(param_groups)

    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=learning_rate,
    )
    return optimizer, aux_optimizer

def compute_aux_loss(aux_list: List, backward=False):
    # a function to compute and backward aux_loss
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()

    return aux_loss_sum

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    # https://blog.csdn.net/qq_30565883/article/details/136344599
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]  # x轴方差
    uncertainty[:, 1] = L[:, 0, 1]  # xy协方差
    uncertainty[:, 2] = L[:, 0, 2]  # xz协方差
    uncertainty[:, 3] = L[:, 1, 1]  # y轴方差
    uncertainty[:, 4] = L[:, 1, 2]  # yz协方差
    uncertainty[:, 5] = L[:, 2, 2]  # z轴方差
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best=False, log_dir=None, filename="ckpt.pth.tar"):
    save_file_dir = os.path.join(log_dir, filename)
    print("save model in:", save_file_dir)
    torch.save(state, save_file_dir)
    if is_best:
        torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))
