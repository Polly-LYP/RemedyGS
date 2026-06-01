import numpy as np
import os
import torch

from plyfile import PlyData
# from utils.logger import get_root_logger, print_log
from .io import IO
import math 
from torch.utils.data import Dataset

def load_ply_data_numpy(path, device="cuda"):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    N_Gaussian = xyz.shape[0]
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3   # We are using sh_degree = 3, so
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # gaussian_features_numpy = np.concatenate([features_dc.astype(np.float32).reshape(N_Gaussian, -1), 
    #                                     features_extra.astype(np.float32).reshape(N_Gaussian, -1),
    #                                     opacities, scales.astype(np.float32), rots.astype(np.float32)], axis=1)
    gaussian_features_numpy = torch.cat([torch.from_numpy(features_dc).float().view(N_Gaussian, -1), 
                                        torch.from_numpy(features_extra).float().view(N_Gaussian, -1),
                                        torch.from_numpy(opacities).float(), 
                                        torch.from_numpy(scales).float(), 
                                        torch.from_numpy(rots).float()], axis=1)
    assert gaussian_features_numpy.shape[1] == 56
    
    return xyz, gaussian_features_numpy


class DL3DV_GSnIG_Zn(Dataset):
    def __init__(self, gs_dataset_path, dataset_args, dataset_list, subset="train"):
        print(f"Gaussian Dataset Path: {gs_dataset_path} \n Image Dataset Path {dataset_args.source_path} \n Dataset list: {dataset_list}")
        self.gs_path = gs_dataset_path
        self.dataset_args = dataset_args
        assert subset in ["train", "test"]
        self.subset = subset
        self.data_list_file = dataset_list
        # test_data_list_file = os.path.join(self.data_root, "test.txt")
        print_log(f"[DATASET] Open file {self.data_list_file}", logger="DL3DV-2K")
        
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()
        self.file_list = []
        for line in lines:
            line = line.strip()
            # print("line", line)
            fold_num = line.split("-")[0]
            hash_code = line.split("/")[-1]
            self.file_list.append(
                {"gs_path": line, "image_path": os.path.join(fold_num, hash_code)}
            )
        print_log(f"[DATASET] {len(self.file_list)} instances were loaded", logger="DL3DV-2K",)
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        updated_source_path = os.path.join(self.dataset_args.source_path, sample["image_path"])
        gaussian_positions_numpy, gaussian_feature_numpy = load_ply_data_numpy(os.path.join(self.gs_path, sample["gs_path"], 'point_cloud', 'iteration_30000', 'point_cloud.ply'))
        # 这里还可以放一点预处理，假如后面需要normalization之类的？
        # vertex = gs["vertex"]
        # data = read_gaussian_attribute(vertex, self.attribute)
        # data = torch.from_numpy(data).float()

        return gaussian_positions_numpy, gaussian_feature_numpy, updated_source_path

    def __len__(self):
        return len(self.file_list)


class DL3DV_SimplePath_Zn(Dataset):
    def __init__(self, args, dataset_args, subset="train"):
        # print(f" Gaussian Dataset Path: {gs_dataset_path} \n Image Dataset Path {dataset_args.source_path} \n Dataset list: {dataset_list}")
        self.gs_path = args.GS_dataset_path
        self.dataset_args = dataset_args
        assert subset in ["train", "test"]
        self.subset = subset
        self.data_list_file = args.dataset_list_config
        # test_data_list_file = os.path.join(self.data_root, "test.txt")
        # print(f"[DATASET] Open file {self.data_list_file}")
        
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()
        self.file_list = []
        for line in lines:
            line = line.strip()
            # print("line", line)
            fold_num = line.split("-")[0]
            hash_code = line.split("/")[-1]
            self.file_list.append(
                {"gs_path": line, "image_path": os.path.join(fold_num, hash_code)}
            )
        print_log(f"[DATASET] {len(self.file_list)} instances were loaded", logger="DL3DV-2K",)
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        updated_source_path = os.path.join(self.dataset_args.source_path, sample["image_path"])
        updated_model_path = os.path.join(self.gs_path, sample["gs_path"])

        return updated_model_path, updated_source_path

    def __len__(self):
        return len(self.file_list)


# Function to simulate print_log for debugging
def print_log(message, logger=None):
    print(f"{logger}: {message}")



# def read_gaussian_attribute(vertex, attribute):
#     # assert 'xyz' in attribute, 'At least need xyz attribute' can free this one actually
#     # record the attribute and the index to read it
#     # attribute_index = {}
#     if "xyz" in attribute:
#         x = vertex["x"].astype(np.float32)
#         y = vertex["y"].astype(np.float32)
#         z = vertex["z"].astype(np.float32)
#         data = np.stack((x, y, z), axis=-1)  # [n, 3]

#     if "opacity" in attribute:
#         opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
#         opacity = np_sigmoid(opacity)   # 这个存疑？
#         # opacity range from 0 to 1
#         data = np.concatenate((data, opacity), axis=-1)

#     if "scale" in attribute and "rotation" in attribute:
#         scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
#         scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
#         scales = np.zeros((data.shape[0], len(scale_names)))
#         for idx, attr_name in enumerate(scale_names):
#             scales[:, idx] = vertex[attr_name].astype(np.float32)

#         scales = np.exp(scales)  # scale normalization

#         rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
#         rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
#         rots = np.zeros((data.shape[0], len(rot_names)))
#         for idx, attr_name in enumerate(rot_names):
#             rots[:, idx] = vertex[attr_name].astype(np.float32)

#         rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
#         # always set the first to be positive
#         signs_vector = np.sign(rots[:, 0])
#         rots = rots * signs_vector[:, None]

#         data = np.concatenate((data, scales, rots), axis=-1)

#     if "sh" in attribute:
#         # get 3 dimension of sphere homrincals
#         features_dc = np.zeros((data.shape[0], 3, 1))
#         features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
#         features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
#         features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

#         feature_pc = features_dc.reshape(-1, 3)
#         data = np.concatenate((data, feature_pc), axis=1)

#     return data