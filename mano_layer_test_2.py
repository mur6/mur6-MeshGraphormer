import argparse
import itertools
import os.path
from collections import namedtuple
from pathlib import Path

# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
# from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.modeling._mano import MANO
from manopth.manolayer import ManoLayer
#from src.datasets.hand_mesh_tsv import HandMeshTSVDataset, HandMeshTSVYamlDataset
from src.datasets.my_dataset import BlenderHandMeshDataset



import argparse
import datetime
import gc
import json
import os
import os.path as op
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.utils import make_grid

import src.modeling.data.config as cfg
from src.datasets.build import make_batch_data_sampler, make_data_sampler
from src.datasets.my_dataset import BlenderHandMeshDataset
from src.modeling._mano import MANO, Mesh
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.utils.comm import all_gather, get_rank, get_world_size, is_main_process, synchronize
from src.utils.geometric_layers import orthographic_projection
from src.utils.logger import setup_logger
from src.utils.metric_logger import AverageMeter
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.renderer import (
    Renderer,
    visualize_reconstruction,
    visualize_reconstruction_no_text,
    visualize_reconstruction_test,
)


def show_3d_plot(axs, points3d_1, points3d_2):
    # print(pred_v3d.shape, pred_v3d)
    for i, points3d in enumerate((points3d_1, points3d_2)):
        points3d /= 164.0
        X, Y, Z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
        # axs.scatter(X, Y, Z, alpha=0.1)
        if i == 0:
            axs.scatter(X, Y, Z, alpha=0.1)
        else:
            axs.scatter(X, Y, Z, color='r')
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() * 0.5
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    axs.set_xlim(mid_x - max_range, mid_x + max_range)
    axs.set_ylim(mid_y - max_range, mid_y + max_range)
    axs.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_data_3d(gt_vertices_sub, gt_3d_joints):
    # torch.set_printoptions(precision=2)
    # print("gt_vertices_sub.shape:", gt_vertices_sub.shape)
    # print("gt_3d_joints.shape:", gt_3d_joints.shape)
    verts = gt_vertices_sub[0]
    joints = gt_3d_joints[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][batch_idx]
    #if mano_faces is None:
    show_3d_plot(ax, verts, joints)
    # ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    plt.show()


def build_hand_dataset(yaml_file, args, is_train=True, scale_factor=1):
    if not os.path.isfile(yaml_file):
        yaml_file = os.path.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert os.path.isfile(yaml_file)
    return HandMeshTSVYamlDataset(args, yaml_file, is_train, False, scale_factor)


MetaInfo = namedtuple("MetaInfo", "pose,betas,joints_2d,joints_3d")


def iter_meta_info(dataset_partial):
    for img_key, transfromed_img, meta_data in dataset_partial:
        pose = meta_data["pose"]
        assert pose.shape == (48,)
        # mano_pose, trans = pose[:45], pose[45:]
        # assert mano_pose.shape == (45,)
        # assert trans.shape == (3,)
        betas = meta_data["betas"]
        assert betas.shape == (10,)
        joints_2d = meta_data["joints_2d"][:, 0:2]
        assert joints_2d.shape == (21, 2)
        joints_3d = meta_data["joints_3d"][:, 0:3]
        assert joints_3d.shape == (21, 3)
        # print(mano_pose.shape, trans.shape, betas.shape, joints_2d.shape, joints_3d.shape)
        yield MetaInfo(pose, betas, joints_2d, joints_3d)



def make_hand_data_loader(
    args, *, blender_ds_base_path, is_distributed=True, is_train=True, start_iter=0, scale_factor=1
):
    dataset = BlenderHandMeshDataset(base_path=blender_ds_base_path)
    shuffle = True
    images_per_gpu = args.per_gpu_train_batch_size
    images_per_batch = images_per_gpu * get_world_size()
    # print(f"images_per_batch: {images_per_batch}")
    # print(f"dataset count: {len(dataset)}")
    iters_per_batch = len(dataset) // images_per_batch
    print(f"iters_per_batch: {iters_per_batch}")
    num_iters = iters_per_batch * args.num_train_epochs
    # logger.info("Train with {} images per GPU.".format(images_per_gpu))
    # logger.info("Total batch size {}".format(images_per_batch))
    # logger.info("Total training steps {}".format(num_iters))

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(sampler, images_per_gpu, num_iters, start_iter)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader

def main(args, *, train_dataloader, num):
    train_dataloader
    # meta_info = list(iter_meta_info(itertools.islice(dataset, num)))[0]
    # # print(meta_info)
    # mano_model = MANO().to("cpu")
    # # mano_model.layer = mano_model.layer.cuda()
    # mano_layer = mano_model.layer
    # pose = meta_info.pose.unsqueeze(0)
    # betas = meta_info.betas.unsqueeze(0)
    # gt_vertices, gt_3d_joints = mano_model.layer(pose, betas)

    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # #verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][batch_idx]
    # # #if mano_faces is None:
    # visualize_data_3d(gt_vertices, gt_3d_joints)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiscale_inference",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--train_yaml",
        type=Path,
        required=True,
        help="Yaml file with all data for training.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        # required=True,
    )
    parser.add_argument(
        "--blender_ds_base_path",
        type=Path,
        required=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    blender_ds_base_path = args.blender_ds_base_path
    train_dataloader = make_hand_data_loader(
        args,
        blender_ds_base_path=blender_ds_base_path,
        is_distributed=False,
        is_train=True,
        scale_factor=args.img_scale_factor,
    )
    main(args, train_dataloader, num=args.num)
