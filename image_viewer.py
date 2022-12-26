import argparse
import itertools
from collections import namedtuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

import src.modeling.data.config as cfg
from manopth.manolayer import ManoLayer
from src.datasets.build import make_batch_data_sampler, make_data_sampler

# from src.datasets.hand_mesh_tsv import HandMeshTSVDataset, HandMeshTSVYamlDataset
from src.datasets.my_dataset import BlenderHandMeshDataset
from src.modeling._mano import MANO, Mesh
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.utils.comm import all_gather, get_rank, get_world_size, is_main_process, synchronize







def visualize_data(image, ori_img, joints_2d, mano_pose=None, shape=None):
    n_cols = 2
    n_rows = 2
    fig, axs = plt.subplots(n_cols, n_rows, figsize=(9, 9))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title("joints_2d")
    ax.imshow(image)
    # print(joints_2d)
    # # joints_2d = joints_2d * img_size
    # print(f"joints_2d: {joints_2d}")
    # # joints_2d = ((joints_2d[:, :2] + 1) * 0.5) * img_size
    # print(joints_2d)
    ax.scatter(joints_2d[:, 0], joints_2d[:, 1], c="red", alpha=0.75)
    ax = axs[1]
    ax.set_title("ori_img")
    ax.imshow(ori_img)

    # ax = axs[2]
    # ax.set_title("shape[10]")
    # ax.plot(shape)
    # start, end = ax.get_xlim()
    # ax.xaxis.set_ticks(np.arange(0.0, 10.0, 1.0))
    # ax.yaxis.set_ticks(np.arange(-1.5, 3.0, 0.25))
    # ax.grid()
    plt.tight_layout()
    plt.show()


# def make_hand_data_loader(
#     args, *, blender_ds_base_path, is_distributed=True, is_train=True, start_iter=0, scale_factor=1
# ):
#     dataset = BlenderHandMeshDataset(base_path=blender_ds_base_path)
#     shuffle = True
#     images_per_gpu = args.per_gpu_train_batch_size
#     images_per_batch = images_per_gpu * get_world_size()
#     # print(f"images_per_batch: {images_per_batch}")
#     # print(f"dataset count: {len(dataset)}")
#     iters_per_batch = len(dataset) // images_per_batch
#     print(f"iters_per_batch: {iters_per_batch}")
#     num_iters = iters_per_batch * args.num_train_epochs
#     # logger.info("Train with {} images per GPU.".format(images_per_gpu))
#     # logger.info("Total batch size {}".format(images_per_batch))
#     # logger.info("Total training steps {}".format(num_iters))

#     sampler = make_data_sampler(dataset, shuffle, is_distributed)
#     batch_sampler = make_batch_data_sampler(sampler, images_per_gpu, num_iters, start_iter)
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         num_workers=args.num_workers,
#         batch_sampler=batch_sampler,
#         pin_memory=True,
#     )
#     return data_loader

def iter_images(dataset, num):
    dataloader = torch.utils.data.DataLoader(dataset)
    dataloder_it = itertools.islice(dataloader, num)
    for img_keys, images, annotations in dataloder_it:
        img = images[0]
        img = img.numpy().transpose(1, 2, 0)
        yield img


def visualize_image_data(images, num):
    n_cols = 4
    n_rows = 5
    fig, axses = plt.subplots(n_cols, n_rows, figsize=(9, 9))
    for ax, image in zip(axses.flatten(), images):
        ax.imshow(image)
    plt.tight_layout()
    plt.show()

def main(dataset, num):
    images = list(iter_images(dataset, num))
    visualize_image_data(images, num)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiscale_inference",
        default=False,
        action="store_true",
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
    dataset = BlenderHandMeshDataset(base_path=blender_ds_base_path)
    main(dataset, num=args.num)
