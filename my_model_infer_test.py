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

from logging import (DEBUG, INFO, basicConfig, critical, debug, error,
                     exception, info, getLogger)

# python -m torch.distributed.launch --nproc_per_node=8 \
#        src/tools/run_gphmer_handmesh.py \
#        --train_yaml freihand/train.yaml \
#        --val_yaml freihand/test.yaml \
#        --arch hrnet-w64 \
#        --num_workers 4 \
#        --per_gpu_train_batch_size 32 \
#        --per_gpu_eval_batch_size 32 \
#        --num_hidden_layers 4 \
#        --num_attention_heads 4 \
#        --lr 1e-4 \
#        --num_train_epochs 200 \
#        --input_feat_dim 2051,512,128 \
#        --hidden_feat_dim 1024,256,64


def main_1(args):
    device = "cpu"
    mano_model = MANO().to(device)


def main_2(args):
    basicConfig(level=INFO)
    logger = getLogger(__name__)
    device = "cpu"
    mano_model = MANO().to(device)


    # Load pretrained model
    trans_encoder = []

    input_feat_dim = "2051,512,128"
    hidden_feat_dim = "1024,256,64"
    input_feat_dim = [int(item) for item in input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

    # which encoder block to have graph convs
    which_gcn='0,0,1'
    which_blk_graph = [int(item) for item in which_gcn.split(',')]

    # if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
    #     # if only run eval, load checkpoint
    #     logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
    #     _model = torch.load(args.resume_checkpoint)


    # init three transformer-encoder blocks in a loop
    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, Graphormer
        model_name_or_path = 'src/modeling/bert/bert-base-uncased/'
        config = config_class.from_pretrained(model_name_or_path)
        config.output_attentions = False
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        hidden_size = hidden_feat_dim[i]
        intermediate_size = int(hidden_size*2)

        if which_blk_graph[i]==1:
            config.graph_conv = True
            logger.info("Add Graph Conv")
        else:
            config.graph_conv = False

        config.mesh_type = "hand"

        # # update model structure if specified in arguments
        # update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
        # for idx, param in enumerate(update_params):
        #     arg_param = getattr(args, param)
        #     config_param = getattr(config, param)
        #     if arg_param > 0 and arg_param != config_param:
        #         logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
        #         setattr(config, param, arg_param)

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config)
        logger.info("Init model from scratch.")
        trans_encoder.append(model)

        # create backbone model
        # default arch is `hrnet-w64`
        hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        logger.info('=> loading... hrnet-v2-w64 model')
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loaded hrnet-v2-w64 model')

        trans_encoder_first = trans_encoder[0]
        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        _model = Graphormer_Network(args, config, backbone, trans_encoder)


    # update configs to enable attention outputs
    setattr(_model.trans_encoder[-1].config,'output_attentions', True)
    setattr(_model.trans_encoder[-1].config,'output_hidden_states', True)
    _model.trans_encoder[-1].bert.encoder.output_attentions = True
    _model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_model.trans_encoder[-1].config, 'device', device)

    _model.to(device)
    logger.info("Model Loaded.")


if __name__ == "__main__":
    #args = parse_args()
    main_2(args=None)
