import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# from src.modeling.hrnet.config import update_config as hrnet_update_config
# from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
# from src.utils.comm import all_gather, get_rank, get_world_size, is_main_process, synchronize


HandMeta = namedtuple("HandMeta", "pose betas scale joints_2d, joints_3d")


def add_ones_column(x):
    additional = torch.ones((x.shape[0], 1))
    return torch.cat([x, additional], dim=1)


def get_sorted_files(folder, *, extension):
    iter = folder.glob(f"*.{extension}")
    return list(sorted(iter))


class BlenderHandMeshDataset(object):
    def __init__(self, base_path, scale_factor=1):
        self.base_path = base_path
        self.meta_filepath = base_path / "datageneration/tmp/meta/"
        self.image_filepath = base_path / "datageneration/tmp/rgb/"
        meta_files = get_sorted_files(self.meta_filepath, extension="pkl")
        im_files = get_sorted_files(self.image_filepath, extension="jpg")
        assert len(meta_files) == len(im_files)
        self.data_length = len(meta_files)
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.data_length

    def get_image(self, image_file):
        image = Image.open(image_file)
        return torchvision.transforms.functional.to_tensor(image)

    def get_annotations(self, meta_file):
        b = meta_file.read_bytes()
        d = pickle.loads(b)
        mano_pose = d["mano_pose"]
        trans = d["trans"]
        pose = np.concatenate([mano_pose, trans])
        pose = torch.from_numpy(pose)
        betas = torch.from_numpy(d["shape"])
        scale = d["z"]
        coords_2d = d["coords_2d"]
        joints_2d = torch.from_numpy(coords_2d)
        joints_2d = add_ones_column(joints_2d)
        coords_3d = d["coords_3d"]
        joints_3d = torch.from_numpy(coords_3d)
        joints_3d = add_ones_column(joints_3d)
        return HandMeta(pose, betas, scale, joints_2d, joints_3d)

    def get_metadata_dict(self):
        meta_data = dict(
            center=torch.tensor([112.0, 112]),
            has_2d_joints=1,
            has_3d_joints=1,
            has_smpl=1,
            mjm_mask=torch.ones([21, 1]),
            mvm_mask=torch.ones([195, 1]),
        )
        # print("center: ", center)
        # print(f"mjm_mask: {mjm_mask.shape}")
        # print(f"mvm_mask: {mvm_mask.shape}")
        return meta_data

    def __getitem__(self, idx):
        meta_file = self.meta_filepath / f"{idx:08}.pkl"
        # print(meta_file, meta_file.exists())
        image_file = self.image_filepath / f"{idx:08}.jpg"
        # print(image_file, image_file.exists())

        hand_meta = self.get_annotations(meta_file)
        img = self.get_image(image_file)
        transfromed_img = self.normalize_img(img)
        img_key = image_file.name
        ##############################################
        meta_data = self.get_metadata_dict()
        meta_data["ori_img"] = img
        meta_data["pose"] = hand_meta.pose
        meta_data["betas"] = hand_meta.betas
        meta_data["joints_3d"] = hand_meta.joints_3d
        meta_data["joints_2d"] = hand_meta.joints_2d
        meta_data["scale"] = hand_meta.scale
        return (img_key, transfromed_img, meta_data)
