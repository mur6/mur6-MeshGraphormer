"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function

import argparse
import base64
import code
import datetime
import gc
import json
import os
import os.path as op
import pickle
import sys
import time
from collections import defaultdict
from logging import DEBUG, INFO, basicConfig, critical, debug, error, exception, getLogger, info
from os.path import join

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

# from metro.modeling._mano import MANO
# mano_mesh_model = MANO() # init MANO
import scipy.misc
import torch
import torchvision.models as models
import transforms3d
from pycocotools.coco import COCO
from torchvision.utils import make_grid
from tqdm import tqdm

import src.modeling.data.config as cfg
from my_model_tools import get_mano_model, get_model_for_train
from src.datasets.build import make_hand_data_loader
from src.datasets.hand_mesh_tsv import HandMeshTSVDataset, HandMeshTSVYamlDataset
from src.datasets.human_mesh_tsv import MeshTSVDataset, MeshTSVYamlDataset
from src.modeling._mano import MANO, Mesh
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.utils.comm import all_gather, get_rank, get_world_size, is_main_process, synchronize
from src.utils.geometric_layers import orthographic_projection
from src.utils.image_ops import img_from_base64
from src.utils.logger import setup_logger
from src.utils.metric_logger import AverageMeter
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.tsv_file import TSVFile
from src.utils.tsv_file_ops import generate_hw_file, generate_linelist_file, tsv_reader, tsv_writer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiscale_inference",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return args


def build_hand_dataset(yaml_file, args, is_train=True, scale_factor=1):
    print(yaml_file)
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert op.isfile(yaml_file)
    return HandMeshTSVYamlDataset(args, yaml_file, is_train, False, scale_factor)


def make_hand_data_loader(args, yaml_file, is_distributed=True, is_train=True, start_iter=0, scale_factor=1):

    dataset = build_hand_dataset(yaml_file, args, is_train=is_train, scale_factor=scale_factor)
    logger = logging.getLogger(__name__)
    if is_train == True:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(sampler, images_per_gpu, num_iters, start_iter)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader


def main():
    device = torch.device("cpu")
    train_yaml_file = "../orig-MeshGraphormer/freihand/train.yaml"
    img_scale_factor = 1
    # train_dataloader = make_hand_data_loader(
    #     None,
    #     train_yaml,
    #     is_distributed=False,
    #     is_train=True,
    #     scale_factor=img_scale_factor,
    # )
    args = parse_args()
    dataset = build_hand_dataset(train_yaml_file, args, is_train=True)
    for i in range(10):
        img_key, transfromed_img, meta_data = dataset[i]
        print("######################")
        for name, value in meta_data.items():
            if name in ("center", "has_smpl"):
                print(f"{name} val={value}")
            elif hasattr(value, "shape"):
                print(f"{name}\t{value.shape}")
            else:
                print(f"{name} val={value}")
        print("######################")
        # mjm_mask = meta_data["mjm_mask"]
        # print(mjm_mask.unsqueeze(0).expand(-1, -1, 2051).shape)
        # mvm_mask = meta_data["mvm_mask"]
        # print(mvm_mask.unsqueeze(0).expand(-1, -1, 2051))
        # mjm_mask_ = mjm_mask.unsqueeze(0).expand(-1, -1, 2051)
        # mvm_mask_ = mvm_mask.unsqueeze(0).expand(-1, -1, 2051)
        # meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)
        # print(meta_masks.shape)
        pose = meta_data["pose"]
        print(f"pose: {pose[:3]} ... {pose[-3:]}")
        betas = meta_data["betas"]
        print(f"betas: {betas}")
        joints_3d = meta_data["joints_3d"]
        # print(f"joints_3d: {joints_3d[0]}")
        print(f"joints_3d: {joints_3d}")
        joints_2d = meta_data["joints_2d"]
        # print(f"joints_2d: {joints_2d[0]}")
        print(f"joints_2d: {joints_2d}")
        break
    # images_per_gpu = 1  # per_gpu_train_batch_size
    # images_per_batch = images_per_gpu * get_world_size()
    # iters_per_batch = len(dataset) // images_per_batch
    # num_iters = iters_per_batch * num_train_epochs
    # logger.info("Train with {} images per GPU.".format(images_per_gpu))
    # logger.info("Total batch size {}".format(images_per_batch))
    # logger.info("Total training steps {}".format(num_iters))


if __name__ == "__main__":
    # args = parse_args()
    main()
