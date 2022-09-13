import argparse
import logging
from logging import getLogger, basicConfig

from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from PIL import Image
import torch
from torchvision import transforms

basicConfig(level=logging.INFO)
logger = getLogger(__name__)

transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
def main():
    hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
    logger.info('=> loading hrnet-v2-w64 model')
    #print(args.arch)
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    logger.info('Backbone total parameters: {}'.format(backbone_total_params))
    image_list = []
    image_file_or_path ="./samples/hand"
    import os
    #     raise ValueError("image_file_or_path not specified")
    # if op.isfile(args.image_file_or_path):
    #     image_list = [args.image_file_or_path]
    for filename in os.listdir(image_file_or_path):
        if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
            image_list.append(image_file_or_path+'/'+filename)
    #print(mano_model)
    print(image_list)
    img = Image.open('./samples/hand/internet_fig4.jpg')
    img_tensor = transform(img)
    print(img_tensor.shape)
    batch_imgs = torch.unsqueeze(img_tensor, 0)
    print(batch_imgs.shape)
    import src.modeling.data.config as cfg
    j_name_index_wrist = cfg.J_NAME.index('Wrist')
    image_feat, grid_feat = backbone(batch_imgs)
    print(grid_feat.shape)

if __name__ == "__main__":
    main()
