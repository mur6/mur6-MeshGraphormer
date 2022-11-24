from logging import (
    DEBUG,
    INFO,
    basicConfig,
    critical,
    debug,
    error,
    exception,
    getLogger,
    info,
)

import torch
import torchvision.models as models
from torchvision.utils import make_grid

import src.modeling.data.config as cfg
from src.modeling._mano import MANO, Mesh
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat

logger = getLogger(__name__)


def get_model(device):
    mano_model = MANO().to(device)

    # Load pretrained model
    trans_encoder = []

    input_feat_dim = "2051,512,128"
    hidden_feat_dim = "1024,256,64"
    input_feat_dim = [int(item) for item in input_feat_dim.split(",")]
    hidden_feat_dim = [int(item) for item in hidden_feat_dim.split(",")]
    output_feat_dim = input_feat_dim[1:] + [3]

    # which encoder block to have graph convs
    which_gcn = "0,0,1"
    which_blk_graph = [int(item) for item in which_gcn.split(",")]

    # init three transformer-encoder blocks in a loop
    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, Graphormer
        model_name_or_path = "src/modeling/bert/bert-base-uncased/"
        config = config_class.from_pretrained(model_name_or_path)
        config.output_attentions = False
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        hidden_size = hidden_feat_dim[i]
        intermediate_size = int(hidden_size * 2)

        if which_blk_graph[i] == 1:
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
        hrnet_yaml = "models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
        hrnet_checkpoint = "models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
        hrnet_update_config(hrnet_config, hrnet_yaml)
        logger.info("=> loading... hrnet-v2-w64 model")
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info("=> loaded hrnet-v2-w64 model")

        trans_encoder_first = trans_encoder[0]
        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info("Graphormer encoders total parameters: {}".format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info("Backbone total parameters: {}".format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        args = None
        _model = Graphormer_Network(args, config, backbone, trans_encoder)

    # update configs to enable attention outputs
    setattr(_model.trans_encoder[-1].config, "output_attentions", True)
    setattr(_model.trans_encoder[-1].config, "output_hidden_states", True)
    _model.trans_encoder[-1].bert.encoder.output_attentions = True
    _model.trans_encoder[-1].bert.encoder.output_hidden_states = True
    for iter_layer in range(4):
        _model.trans_encoder[-1].bert.encoder.layer[
            iter_layer
        ].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_model.trans_encoder[-1].config, "device", device)

    _model.to(device)
    return _model
