"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
from src.modeling.bert import BertConfig, Graphormer
# from src.modeling.bert.modeling_graphormer import EncoderBlock2
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
#from src.utils.renderer import Renderer, visualize_reconstruction_and_att_local, visualize_reconstruction_no_text
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])

import torch.onnx
from src.modeling.bert.e2e_hand_network import generate_t_pose_template_mesh, get_template_vertices_sub, template_normalize


def calc_features(Graphormer_model, images, template_3d_joints, template_vertices_sub):
    batch_size = images.size(0)
    num_joints = template_3d_joints.shape[1]
    # concatinate template joints and template vertices, and then duplicate to batch size
    ref_vertices = torch.cat([template_3d_joints, template_vertices_sub],dim=1)
    ref_vertices = ref_vertices.expand(batch_size, -1, -1)

    # extract grid features and global image features using a CNN backbone
    image_feat, grid_feat = Graphormer_model.backbone(images)
    # concatinate image feat and mesh template
    image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
    # process grid features
    grid_feat = torch.flatten(grid_feat, start_dim=2)
    grid_feat = grid_feat.transpose(1,2)
    grid_feat = Graphormer_model.grid_feat_dim(grid_feat)
    # concatinate image feat and template mesh to form the joint/vertex queries
    features = torch.cat([ref_vertices, image_feat], dim=2)
    # prepare input tokens including joint/vertex queries and grid features
    features = torch.cat([features, grid_feat],dim=1)
    return features


def make_input_ids_and_position_ids():
    batch_size=1
    seq_length=265
    print(f"EncoderBlock: batch_size={batch_size}")
    print(f"EncoderBlock: seq_length={seq_length}")
    input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    print(f"EncoderBlock: input_ids={input_ids.shape}")
    print(f"EncoderBlock: position_ids={position_ids.shape}")
    return input_ids, position_ids


def run_inference(args, image_list, Graphormer_model, mano, trans_encoder):
    mesh_sampler = Mesh()
    # switch to evaluate mode
    Graphormer_model.eval()
    mano.eval()
    with torch.no_grad():
        for image_file in image_list:
            if 'pred' not in image_file:
                att_all = []
                print(image_file)
                img = Image.open(image_file)
                img_tensor = transform(img)
                img_visual = transform_visualize(img)

                batch_imgs = torch.unsqueeze(img_tensor, 0)
                template_vertices, template_3d_joints = generate_t_pose_template_mesh(mano)
                template_vertices_sub = get_template_vertices_sub(mesh_sampler, template_vertices)
                template_vertices, template_3d_joints, template_vertices_sub = template_normalize(template_vertices, template_3d_joints, template_vertices_sub)
                # forward-pass
                #[1]
                pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = Graphormer_model(batch_imgs, template_vertices, template_3d_joints, template_vertices_sub)
                torch.onnx.export(Graphormer_model, (batch_imgs, template_vertices, template_3d_joints, template_vertices_sub), "gm.onnx", opset_version=11)
                return 
                print("#################################\n")
                img_feats = calc_features(Graphormer_model, batch_imgs, template_3d_joints, template_vertices_sub)
                #input_ids, position_ids = make_input_ids_and_position_ids()
                trans_encoder(img_feats, )
                torch.onnx.export(trans_encoder, (img_feats, ), "trans_encoder3.onnx", opset_version=11)
                return 
                # obtain 3d joints from full mesh
                pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
                pred_3d_pelvis = pred_3d_joints_from_mesh[:,cfg.J_NAME.index('Wrist'),:]
                pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]
                pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]
                ##
                template_pose = torch.zeros((1,48))
                template_betas = torch.zeros((1,10))
                template_vertices, template_3d_joints = mano.layer(template_pose, template_betas)
                template_vertices = template_vertices / 1000.0
                template_3d_joints = template_3d_joints / 1000.0
                template_vertices_sub = mesh_sampler.downsample(template_vertices)

                ##############
                img_feats = Graphormer_model.calc_features(batch_imgs, template_vertices, template_3d_joints, template_vertices_sub)
                out = trans_encoder_first(img_feats, input_ids, position_ids)

                break
                # obtain 3d joints from full mesh
                pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
                pred_3d_pelvis = pred_3d_joints_from_mesh[:,cfg.J_NAME.index('Wrist'),:]
                pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]
                pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

                # save attantion
                att_max_value = att[-1]
                att_cpu = np.asarray(att_max_value.cpu().detach())
                att_all.append(att_cpu)

                # obtain 3d joints, which are regressed from the full mesh
                pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
                # obtain 2d joints, which are projected from 3d joints of mesh
                pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
                pred_2d_coarse_vertices_from_mesh = orthographic_projection(pred_vertices_sub.contiguous(), pred_camera.contiguous())


                # visual_imgs_output = visualize_mesh(renderer, batch_visual_imgs[0],
                #                                     pred_vertices[0].detach(),
                #                                     pred_camera.detach())
                # visual_imgs = visual_imgs_output.transpose(1,2,0)

    return #pred_vertices[0], pred_camera

def visualize_mesh( renderer, images,
                    pred_vertices_full,
                    pred_camera):
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full.cpu().numpy() 
    cam = pred_camera.cpu().numpy()
    # Visualize only mesh reconstruction 
    rend_img = visualize_reconstruction_no_text(img, 224, vertices_full, cam, renderer, color='light_blue')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img

def visualize_mesh_and_attention( renderer, images,
                    pred_vertices_full,
                    pred_vertices, 
                    pred_2d_vertices,
                    pred_2d_joints,
                    pred_camera,
                    attention):
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full.cpu().numpy() 
    vertices = pred_vertices.cpu().numpy()
    vertices_2d = pred_2d_vertices.cpu().numpy()
    joints_2d = pred_2d_joints.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    att = attention.cpu().numpy()
    # Visualize reconstruction and attention
    rend_img = visualize_reconstruction_and_att_local(img, 224, vertices_full, vertices, vertices_2d, cam, renderer, joints_2d, att, color='light_blue')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img

def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")       
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.")  
    parser.add_argument("--image_file_or_path", default='./samples/hand', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                        help="The Image Feature Dimension.")  
    parser.add_argument("--which_gcn", default='0,0,1', type=str, 
                        help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv") 
    parser.add_argument("--mesh_type", default='hand', type=str, help="body or hand") 

    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=True, action='store_true',) 
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    args = parser.parse_args()
    return args

def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))
   
    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    #mano_model.layer = mano_model.layer.cuda()
    #mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = None#Renderer(faces=mano_model.face)

    # Load pretrained model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]
    
    # which encoder block to have graph convs
    which_blk_graph = [int(item) for item in args.which_gcn.split(',')]

    if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _model = torch.load(args.resume_checkpoint)

    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(args.config_name if args.config_name \
                    else args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*2)

            if which_blk_graph[i]==1:
                config.graph_conv = True
                logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = args.mesh_type

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)
        
        # create backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        trans_encoder_first = trans_encoder[0]
        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        _model = Graphormer_Network(args, config, backbone, trans_encoder)

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            # workaround approach to load sparse tensor in graph conv.
            state_dict = torch.load(args.resume_checkpoint)
            _model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

    # update configs to enable attention outputs
    setattr(_model.trans_encoder[-1].config,'output_attentions', True)
    setattr(_model.trans_encoder[-1].config,'output_hidden_states', True)
    _model.trans_encoder[-1].bert.encoder.output_attentions = True
    _model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_model.trans_encoder[-1].config,'device', args.device)

    _model.to(args.device)
    logger.info("Run inference")

    image_list = []
    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        # should be a path with images only
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                image_list.append(args.image_file_or_path+'/'+filename)
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))
    print(mano_model)
    print(image_list)
    run_inference(args, image_list, _model, mano_model, trans_encoder)

if __name__ == "__main__":
    args = parse_args()
    main(args)
