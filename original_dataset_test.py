"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
from pathlib import Path
import torch
import torchvision.models as models
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build import make_hand_data_loader

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter
from src.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test, visualize_reconstruction_no_text
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection






def visualize_mesh( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(21))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def visualize_mesh_test( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices,
                    pred_camera,
                    pred_keypoints_2d,
                    PAmPJPE):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(21))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        score = PAmPJPE[i]
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction_test(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer, score)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img)) 
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def visualize_mesh_no_text(
        renderer,
        images,
        pred_vertices,
        pred_camera):
    """Tensorboard logging."""
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 1)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction only
        rend_img = visualize_reconstruction_no_text(img, 224, vertices, cam, renderer, color='hand')
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs


# def main2(args):
#     global logger
#     # Setup CUDA, GPU & distributed training
#     args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
#     os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
#     print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))
    
#     args.distributed = args.num_gpus > 1
#     args.device = torch.device(args.device)
#     if args.distributed:
#         print("Init distributed training on local rank {}".format(args.local_rank))
#         torch.cuda.set_device(args.local_rank)
#         torch.distributed.init_process_group(
#             backend='nccl', init_method='env://'
#         )
#         synchronize()
   
#     mkdir(args.output_dir)
#     logger = setup_logger("Graphormer", args.output_dir, get_rank())
#     set_seed(args.seed, args.num_gpus)
#     logger.info("Using {} GPUs".format(args.num_gpus))

#     # Mesh and SMPL utils
#     mano_model = MANO().to(args.device)
#     mano_model.layer = mano_model.layer.cuda()
#     mesh_sampler = Mesh()

#     # Renderer for visualization
#     renderer = Renderer(faces=mano_model.face)

#     # Load pretrained model
#     trans_encoder = []

#     input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
#     hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
#     output_feat_dim = input_feat_dim[1:] + [3]

#     # which encoder block to have graph convs
#     which_blk_graph = [int(item) for item in args.which_gcn.split(',')]

#     if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
#         # if only run eval, load checkpoint
#         logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
#         _model = torch.load(args.resume_checkpoint)

#     else:
#         # init three transformer-encoder blocks in a loop
#         for i in range(len(output_feat_dim)):
#             config_class, model_class = BertConfig, Graphormer
#             config = config_class.from_pretrained(args.config_name if args.config_name \
#                     else args.model_name_or_path)

#             config.output_attentions = False
#             config.hidden_dropout_prob = args.drop_out
#             config.img_feature_dim = input_feat_dim[i]
#             config.output_feature_dim = output_feat_dim[i]
#             args.hidden_size = hidden_feat_dim[i]
#             args.intermediate_size = int(args.hidden_size*2)

#             if which_blk_graph[i]==1:
#                 config.graph_conv = True
#                 logger.info("Add Graph Conv")
#             else:
#                 config.graph_conv = False

#             config.mesh_type = args.mesh_type

#             # update model structure if specified in arguments
#             update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
#             for idx, param in enumerate(update_params):
#                 arg_param = getattr(args, param)
#                 config_param = getattr(config, param)
#                 if arg_param > 0 and arg_param != config_param:
#                     logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
#                     setattr(config, param, arg_param)

#             # init a transformer encoder and append it to a list
#             assert config.hidden_size % config.num_attention_heads == 0
#             model = model_class(config=config) 
#             logger.info("Init model from scratch.")
#             trans_encoder.append(model)
        
#         # create backbone model
#         if args.arch=='hrnet':
#             hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
#             hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
#             hrnet_update_config(hrnet_config, hrnet_yaml)
#             backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
#             logger.info('=> loading hrnet-v2-w40 model')
#         elif args.arch=='hrnet-w64':
#             hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
#             hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
#             hrnet_update_config(hrnet_config, hrnet_yaml)
#             backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
#             logger.info('=> loading hrnet-v2-w64 model')
#         else:
#             print("=> using pre-trained model '{}'".format(args.arch))
#             backbone = models.__dict__[args.arch](pretrained=True)
#             # remove the last fc layer
#             backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

#         trans_encoder = torch.nn.Sequential(*trans_encoder)
#         total_params = sum(p.numel() for p in trans_encoder.parameters())
#         logger.info('Graphormer encoders total parameters: {}'.format(total_params))
#         backbone_total_params = sum(p.numel() for p in backbone.parameters())
#         logger.info('Backbone total parameters: {}'.format(backbone_total_params))

#         # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
#         _model = Graphormer_Network(args, config, backbone, trans_encoder)

#         if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
#             # for fine-tuning or resume training or inference, load weights from checkpoint
#             logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
#             # workaround approach to load sparse tensor in graph conv.
#             state_dict = torch.load(args.resume_checkpoint)
#             _model.load_state_dict(state_dict, strict=False)
#             del state_dict
#             gc.collect()
#             torch.cuda.empty_cache()

def visualize_data(image, ori_img, coords_2d=None, mano_pose=None, shape=None):
    n_cols = 2
    n_rows = 2
    fig, axs = plt.subplots(n_cols, n_rows, figsize=(9, 9))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title("coords_2d")
    ax.imshow(image)
    # ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c="red", alpha=0.75)
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


def main(args, *, train_yaml_file, num):
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    # print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))
    args.distributed = args.num_gpus > 1
    # args.distributed = False
    print(f"args.distributed: {args.distributed}")
    train_dataloader = make_hand_data_loader(
        args, args.train_yaml,
        args.distributed,
        is_train=True,
        scale_factor=args.img_scale_factor)
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    print(f"iters_per_epoch: {iters_per_epoch}")

    for iteration, (img_keys, images, annotations) in enumerate(train_dataloader):
        #epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        print(f"batch_size: {batch_size}")

        # images = images.cuda()
        gt_2d_joints = annotations['joints_2d']
        gt_pose = annotations['pose']
        gt_betas = annotations['betas']
        has_mesh = annotations['has_smpl']
        has_3d_joints = has_mesh
        has_2d_joints = has_mesh
        mjm_mask = annotations['mjm_mask']
        mvm_mask = annotations['mvm_mask']
        break
    img = images[0].cpu().numpy().transpose(1,2,0)
    ori_img = annotations['ori_img'][0].numpy().transpose(1,2,0)
    joints_2d = annotations['joints_2d'][0]
    visualize_data(img, ori_img)
    visual_imgs = visualize_mesh(
        renderer,
        pred_vertices.detach(),
        pred_camera.detach(),
        pred_2d_joints_from_mesh.detach())
    # # generate mesh
    # gt_vertices, gt_3d_joints = mano_model.layer(gt_pose, gt_betas)
    # gt_vertices = gt_vertices/1000.0
    # gt_3d_joints = gt_3d_joints/1000.0

    # gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
    # # normalize gt based on hand's wrist
    # gt_3d_root = gt_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
    # gt_vertices = gt_vertices - gt_3d_root[:, None, :]
    # gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
    # gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
    # gt_3d_joints_with_tag = torch.ones((batch_size,gt_3d_joints.shape[1],4)).cuda()
    # gt_3d_joints_with_tag[:,:,:3] = gt_3d_joints

    # # prepare masks for mask vertex/joint modeling
    # mjm_mask_ = mjm_mask.expand(-1,-1,2051)
    # mvm_mask_ = mvm_mask.expand(-1,-1,2051)
    # meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiscale_inference",
        default=False,
        action="store_true",
    )
    parser.add_argument("--img_scale_factor", default=1, type=int, help="adjust image resolution.")
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
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_workers", default=0, type=int, help="Workers in dataloader.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args, train_yaml_file=args.train_yaml, num=args.num)
