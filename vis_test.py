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
from src.utils.renderer import (
    Renderer,
    visualize_reconstruction,
    visualize_reconstruction_no_text,
    visualize_reconstruction_test,
)
from src.utils.tsv_file import TSVFile
from src.utils.tsv_file_ops import generate_hw_file, generate_linelist_file, tsv_reader, tsv_writer


def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, "checkpoint-{}-{}".format(epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, "model.bin"))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, "state_dict.bin"))
            torch.save(args, op.join(checkpoint_dir, "training_args.bin"))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs / 2.0)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss


def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_root = gt_keypoints_3d[:, 0, :]
        gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
        pred_root = pred_keypoints_3d[:, 0, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.0).cuda()


def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.0).cuda()


def visualize_mesh(renderer, images, gt_keypoints_2d, pred_vertices, pred_camera, pred_keypoints_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(21))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2, 0, 1)
        rend_imgs.append(torch.from_numpy(rend_img))
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs


def visualize_mesh_just_image(renderer, images, gt_keypoints_2d):
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(21))
    rend_imgs = []

    img = images.cpu().numpy().transpose(1, 2, 0)

    gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]

    # Get predict vertices for the particular example
    # vertices = pred_vertices[i].cpu().numpy()
    # cam = pred_camera[i].cpu().numpy()

    rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
    rend_img = rend_img.transpose(2, 0, 1)
    rend_imgs.append(torch.from_numpy(rend_img))
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs


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
    # print(yaml_file)
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert op.isfile(yaml_file)
    return HandMeshTSVYamlDataset(args, yaml_file, is_train, False, scale_factor)


def get_dataset():
    # device = torch.device("cpu")
    train_yaml_file = "../orig-MeshGraphormer/freihand/train.yaml"
    # img_scale_factor = 1
    args = parse_args()
    # dataset = build_hand_dataset(train_yaml_file, args, is_train=True)
    # for i in range(10):
    #     img_key, transfromed_img, meta_data = dataset[i]
    #     print(img_key)
    #     print(transfromed_img)
    #     for name, value in meta_data.items():
    #         if hasattr(value, "shape"):
    #             print(f"meta_data: {name} {value.shape}")
    #         else:
    #             print(f"meta_data: {name} {value}")
    #     mjm_mask = meta_data["mjm_mask"]
    #     print(mjm_mask.unsqueeze(0).expand(-1, -1, 2051).shape)
    #     mvm_mask = meta_data["mvm_mask"]
    #     print(mvm_mask.unsqueeze(0).expand(-1, -1, 2051))
    #     mjm_mask_ = mjm_mask.unsqueeze(0).expand(-1, -1, 2051)
    #     mvm_mask_ = mvm_mask.unsqueeze(0).expand(-1, -1, 2051)
    #     meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)
    #     print(meta_masks.shape)
    return build_hand_dataset(train_yaml_file, args, is_train=True)


def main():
    device = torch.device("cpu")
    train_yaml_file = "../orig-MeshGraphormer/freihand/train.yaml"
    # Mesh and SMPL utils
    mano_model = MANO().to(device)
    mano_model.layer = mano_model.layer.cuda()
    # mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=mano_model.face)
    img_key, transfromed_img, annotations = get_dataset()[0]
    annotations.unsqueeze_(0)

    visual_imgs = visualize_mesh(
        renderer,
        annotations["ori_img"].detach(),
        annotations["joints_2d"].detach(),
        pred_vertices.detach(),
        pred_camera.detach(),
        pred_2d_joints_from_mesh.detach(),
    )
