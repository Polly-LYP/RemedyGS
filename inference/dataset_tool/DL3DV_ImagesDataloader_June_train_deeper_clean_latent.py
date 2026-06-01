import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import torchvision.transforms.functional as tf
from torchvision.utils import save_image
import numpy as np
import math
import random
import torch


class DL3DV_MV_Images(Dataset):
    """Dataset class for DL3DV image dataset on sharpnessnet."""
    def __init__(self, img_per_batch=4, root="/home/ylitx/Project/25_4/RemedyGS_Apr/Our_Main/In_domain_exp/Sampled_dataset/100",
                dataset_list='/home/ylitx/Project/25_4/RemedyGS_Apr/Our_Main/In_domain_exp/Sampled_dataset/sampled_scene_id.txt', 
                crop_size=(544,960), subset="train"):
        """
        Args:
            img_per_batch (int): number of images per batch. For each iteration, we aim to train images in between one scene together.
            crop_size (tuple, optional): Size of the crop. Defaults to None. Hope to be full image, so that the 3D geometry is better preserved.
            subset (str, optional): train or test. Defaults to "train".
        """
        super().__init__()
        
        self.path = Path(f"{root}")
        self.subset = subset
        self.img_per_batch = img_per_batch
        self.crop_size = crop_size if subset=="train" else None
        if subset=="train":
            # self.transform = transforms.Compose([transforms.ToTensor(),
            #                 transforms.RandomHorizontalFlip(p=0.05), transforms.RandomVerticalFlip(p=0.05),]) 
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((crop_size[0], crop_size[1]))]) 
            
        else: 
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((crop_size[0], crop_size[1]))])
            self.crop_size = None
        
        self.all_scene_image_list, scene_count, all_img_count = self.get_files(self.path,dataset_list)

        print(f'Loaded dataset from {self.path}, according to {dataset_list}. \n Found {scene_count} scenes, {len(self.all_scene_image_list)} batches, containing {all_img_count} images.')

    def __len__(self):
        return len(self.all_scene_image_list)

    def __getitem__(self, index):
        image_batch = self.all_scene_image_list[index]
        images = [self.transform(Image.open(img).convert('RGB')) for img in image_batch]
        
        if self.crop_size is not None:
            _, h, w = images[0].shape
            th, tw = self.crop_size
            i = (h - th) // 2
            j = (w - tw) // 2
            images = [tf.crop(image, i, j, th, tw) for image in images]
        
        images = torch.stack(images)  # Shape: A*3*H*W



        return images,image_batch
    
    
    def get_files(self,path,dataset_list):
        image_list = []
        img_count = 0
        with open(dataset_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()  # line: hash code
            image_4_dir = Path(os.path.join(path, line,'images_4'))
            if image_4_dir.exists() and image_4_dir.is_dir():
                image_each_hash_list = [str(img_path) for img_path in image_4_dir.glob('*.png')]
                random.shuffle(image_each_hash_list)
                # Split into groups of self.img_per_batch
                for i in range(0, len(image_each_hash_list) - self.img_per_batch + 1, self.img_per_batch):
                    image_list.append(image_each_hash_list[i:i + self.img_per_batch])
                    img_count = img_count + self.img_per_batch
        
        return image_list, len(lines), img_count
    
    
    
    # Version without datalist file
    # def get_files(self):
    #     image_list = []
    #     img_count = 0
    #     for hash_dir in self.path.iterdir():
    #         if hash_dir.is_dir():
    #             image_4_dir = hash_dir / 'images_4'
    #             if image_4_dir.exists() and image_4_dir.is_dir():
    #                 image_each_hash_list = [str(img_path) for img_path in image_4_dir.glob('*.png')]
    #                 random.shuffle(image_each_hash_list)
    #                 # Split into groups of self.img_per_batch
    #                 for i in range(0, len(image_each_hash_list) - self.img_per_batch + 1, self.img_per_batch):
    #                     image_list.append(image_each_hash_list[i:i + self.img_per_batch])
    #                 img_count = img_count + len(image_each_hash_list) - self.img_per_batch + 1
        
    #     return image_list, img_count