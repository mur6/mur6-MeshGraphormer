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


def main():
    device = torch.device("cpu")
    train_yaml = "../orig-MeshGraphormer/freihand/train.yaml"

    train_dataloader = make_hand_data_loader(
        None,
        train_yaml,
        is_distributed=False,
        is_train=True,
        scale_factor=img_scale_factor,
    )


if __name__ == "__main__":
    # args = parse_args()
    main()
