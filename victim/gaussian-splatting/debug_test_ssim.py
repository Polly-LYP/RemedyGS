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
from utils.general_utils import safe_state,PILtoTorch
from utils.loss_utils import l1_loss, ssim

from argparse import ArgumentParser, Namespace

import cv2
import multiprocessing

from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image






defend_image = Image.open("/home/ylitx/Project/25_4/RemedyGS_Apr/Our_Main/baseline/process/processed/Nerf_Synthetic_eps16_smooth/GaussFilter/chair/test/r_0.png")
raw_image = Image.open("/home/ylitx/Project/25_4/RemedyGS_Apr/Main_2/poison-splat/dataset/Nerf_Synthetic/chair/test/r_0.png")
raw_im_data = np.array(raw_image.convert("RGBA"))
raw_white_background = False
raw_bg = np.array([1,1,1]) if raw_white_background else np.array([0, 0, 0])
raw_norm_data = raw_im_data / 255.0
raw_arr = raw_norm_data[:,:,:3] * raw_norm_data[:, :, 3:4] + raw_bg * (1 - raw_norm_data[:, :, 3:4])
raw_image = Image.fromarray(np.array(raw_arr*255.0, dtype=np.byte), "RGB")
raw_resized_image_rgb = PILtoTorch(raw_image, (800,800))
raw_image = raw_resized_image_rgb[:3, ...]

defend_im_data = np.array(defend_image.convert("RGBA"))
defend_white_background = False
defend_bg = np.array([1,1,1]) if defend_white_background else np.array([0, 0, 0])
defend_norm_data = defend_im_data / 255.0
defend_arr = defend_norm_data[:,:,:3] * defend_norm_data[:, :, 3:4] + defend_bg * (1 - defend_norm_data[:, :, 3:4])
defend_image = Image.fromarray(np.array(defend_arr*255.0, dtype=np.byte), "RGB")
defend_resized_image_rgb = PILtoTorch(defend_image, (800,800))
defend_image = defend_resized_image_rgb[:3, ...]


print(ssim(defend_image,raw_image.to(raw_image)).item())