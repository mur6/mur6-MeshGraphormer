import math
import pickle
from collections import namedtuple

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

# from src.modeling.hrnet.config import update_config as hrnet_update_config
# from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
# from src.utils.comm import all_gather, get_rank, get_world_size, is_main_process, synchronize


HandMeta = namedtuple("HandMeta", "pose betas scale joints_2d joints_3d verts_3d")


mean = (0.4917, 0.4626, 0.4153)
std = (0.2401, 0.2368, 0.2520)


noise_t = A.Compose(
    [
        A.Normalize(mean=mean, std=std),
        # A.GaussNoise(var_limit=0.10, p=0.5),
        # A.Blur(blur_limit=7, p=0.5),
        # A.OpticalDistortion(),
        # A.GridDistortion(),
        A.OneOf(
            [
                A.Cutout(num_holes=40, max_h_size=4, max_w_size=4),
                A.Cutout(num_holes=35, max_h_size=8, max_w_size=8),
                A.Cutout(num_holes=20, max_h_size=12, max_w_size=12),
            ],
            p=0.8,
        ),
        ToTensorV2(),
    ]
)


def add_ones_column(x):
    additional = torch.ones((x.shape[0], 1))
    return torch.cat([x, additional], dim=1)


def get_sorted_files(folder, *, extension):
    iter = folder.glob(f"*.{extension}")
    return list(sorted(iter))


class BlenderHandMeshDataset(object):
    def __init__(self, base_path, scale_factor=1, data_length=None):
        self.base_path = base_path
        self.meta_filepath = base_path / "tmp/meta/"
        self.image_filepath = base_path / "tmp/rgb/"
        meta_files = get_sorted_files(self.meta_filepath, extension="pkl")
        im_files = get_sorted_files(self.image_filepath, extension="jpg")
        assert len(meta_files) == len(im_files)
        if data_length is None:
            data_length = len(meta_files)
        else:
            assert data_length <= len(meta_files)
        self.data_length = data_length
        # self.normalize_img = transforms.Normalize(
        #     mean=[0.4917, 0.4626, 0.4153], std=[0.2401, 0.2368, 0.2520]
        # )
        radian = math.pi * (1.42 / 2.0)
        self.rot_mat = torch.FloatTensor(
            [[math.cos(radian), -math.sin(radian), 0.0], [math.sin(radian), math.cos(radian), 0.0], [0.0, 0.0, 1.0]],
        )
        self.rot_mat_2 = torch.diag(torch.FloatTensor([1.0, -1.0, -1.0]))

    def __len__(self):
        return self.data_length

    def get_image(self, image_file):
        image = Image.open(image_file)
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = torchvision.transforms.functional.to_tensor(image)
        transfromed_img = noise_t(image=image)["image"]
        return original_image, transfromed_img

    def adjust_3d_points(self, pts, add_column=False):
        # pts = pts - pts.mean(0)  # @ self.rot_mat
        pts = pts @ self.rot_mat_2
        pts = pts * 1000.0
        if add_column:
            return add_ones_column(pts)
        else:
            return pts

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
        joints_2d = (joints_2d / 112.0) - 1.0
        # print(f"In my dataset joints_2d: {joints_2d}")
        joints_2d = add_ones_column(joints_2d)

        joints_3d = torch.from_numpy(d["coords_3d"])
        # print("mean", joints_3d.mean(0))
        joints_3d = self.adjust_3d_points(joints_3d, add_column=True)

        verts_3d = torch.from_numpy(d["verts_3d"])
        # print("mean", verts_3d.mean(0))
        verts_3d = self.adjust_3d_points(verts_3d)

        # print(f"In my dataset verts_3d[{verts_3d.shape}]:")
        return HandMeta(pose, betas, scale, joints_2d, joints_3d, verts_3d)

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
        img, transfromed_img = self.get_image(image_file)
        img_key = image_file.name
        ##############################################
        meta_data = self.get_metadata_dict()
        meta_data["ori_img"] = img
        meta_data["pose"] = hand_meta.pose
        meta_data["betas"] = hand_meta.betas
        meta_data["joints_3d"] = hand_meta.joints_3d
        meta_data["joints_2d"] = hand_meta.joints_2d
        meta_data["scale"] = hand_meta.scale
        meta_data["verts_3d"] = hand_meta.verts_3d
        return (img_key, transfromed_img, meta_data)
