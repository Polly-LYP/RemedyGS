import numpy as np
import os
import torch

from plyfile import PlyData
from .io import IO
import math 
from torch.utils.data import Dataset


def read_gaussian_attribute(vertex, attribute):
    # assert 'xyz' in attribute, 'At least need xyz attribute' can free this one actually
    # record the attribute and the index to read it
    # attribute_index = {}
    if "xyz" in attribute:
        x = vertex["x"].astype(np.float32)
        y = vertex["y"].astype(np.float32)
        z = vertex["z"].astype(np.float32)
        data = np.stack((x, y, z), axis=-1)  # [n, 3]

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
        opacity = np_sigmoid(opacity)   # 这个存疑？
        # opacity range from 0 to 1
        data = np.concatenate((data, opacity), axis=-1)

    if "scale" in attribute and "rotation" in attribute:
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)

        scales = np.exp(scales)  # scale normalization

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # always set the first to be positive
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]

        data = np.concatenate((data, scales, rots), axis=-1)

    if "sh" in attribute:
        # get 3 dimension of sphere homrincals
        features_dc = np.zeros((data.shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

        feature_pc = features_dc.reshape(-1, 3)
        data = np.concatenate((data, feature_pc), axis=1)

    return data

class ShapeNetGaussian(Dataset):
    def __init__(self, config, subset="train"):
        print("config", config)
        self.data_root = config.DATASPLIT_PATH
        self.gs_path = config.GS_PATH
        assert subset in ["train", "test"]
        self.subset = subset
        self.attribute = config.ATTRIBUTE
        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")
        # test_data_list_file = os.path.join(self.data_root, "test.txt")
        print_log(f"[DATASET] Using Guassian Attribute {self.attribute}", logger="ShapeNetGS-55",)
        print_log(f"[DATASET] Open file {self.data_list_file}", logger="ShapeNetGS-55")
        
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()
        self.file_list = []
        for line in lines:
            line = line.strip()
            # print("line", line)
            taxonomy_id = line.split("-")[0]
            model_id = line.split("-")[1].split(".")[0]
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": line}
            )
        print_log(f"[DATASET] {len(self.file_list)} instances were loaded", logger="ShapeNetGS-55",)
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        try:
            gs = IO.get(os.path.join(self.gs_path, sample["file_path"]))
        except Exception:
            print("Error in loading", os.path.join(self.gs_path, sample["file_path"]))

        vertex = gs["vertex"]
        data = read_gaussian_attribute(vertex, self.attribute)
        data = torch.from_numpy(data).float()

        return sample["taxonomy_id"], sample["model_id"], data  # 因为我们没做normalization，所以不需要返回均值方差

    def __len__(self):
        return len(self.file_list)



# Function to simulate print_log for debugging
def print_log(message, logger=None):
    print(f"{logger}: {message}")