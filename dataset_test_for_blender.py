import argparse
import math
import os.path as op
import pickle
from collections import namedtuple
from logging import DEBUG, INFO, basicConfig, critical, debug, error, exception, getLogger, info
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.models as models
import transforms3d
from PIL import Image
from pycocotools.coco import COCO
from torchvision.utils import make_grid

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


# def visualize_data(*, image, mano_pose, shape):
#     n_cols = 2
#     n_rows = 2
#     fig, axs = plt.subplots(n_cols, n_rows, figsize=(9, 9))
#     axs = axs.flatten()
#     ax = axs[0]
#     ax.set_title("original image")
#     ax.imshow(image)
#     ax = axs[1]
#     ax.set_title("mano_pose[45]")
#     ax.plot(mano_pose)
#     ax.grid()
#     ax = axs[2]
#     ax.set_title("shape[10]")
#     ax.plot(shape)
#     start, end = ax.get_xlim()
#     ax.xaxis.set_ticks(np.arange(0.0, 10.0, 1.0))
#     ax.yaxis.set_ticks(np.arange(-1.5, 3.0, 0.25))
#     ax.grid()
#     plt.tight_layout()
#     plt.show()


def show_data_info(img_key, transfromed_img, meta_data):

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
    betas = meta_data["betas"]
    image = meta_data["ori_img"].detach().cpu().numpy().transpose(1, 2, 0)
    visualize_data(image=image, mano_pose=pose, shape=betas)
    joints_3d = meta_data["joints_3d"]
    joints_2d = meta_data["joints_2d"]
    # print(f"pose: {pose[:3]} ... {pose[-3:]}")
    show_stats(name="pose", value=pose)
    show_stats(name="betas", value=betas)
    # print(f"joints_3d: {joints_3d}")
    # print(f"joints_3d: {joints_3d[:, 0:3]}")
    show_stats(name="joints_3d", value=joints_3d[:, 0:3])
    # print(f"joints_2d: {joints_2d}")
    # print(f"joints_2d: {joints_2d[:, 0:2]}")
    show_stats(name="joints_2d", value=joints_2d[:, 0:2])


def main(*, data_index):
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

    img_key, transfromed_img, meta_data = dataset[data_index]
    show_data_info(img_key, transfromed_img, meta_data)
    # images_per_gpu = 1  # per_gpu_train_batch_size
    # images_per_batch = images_per_gpu * get_world_size()
    # iters_per_batch = len(dataset) // images_per_batch
    # num_iters = iters_per_batch * num_train_epochs
    # logger.info("Train with {} images per GPU.".format(images_per_gpu))
    # logger.info("Total batch size {}".format(images_per_batch))
    # logger.info("Total training steps {}".format(num_iters))


def show_stats(*, name, value):
    var, mean = torch.var_mean(value)
    st = math.sqrt(var)
    print(f"{name}[{value.shape}] => mean:{mean:.2f} std:{st:.2f} var:{var:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, required=True)
    # parser.add_argument(
    #     "--multiscale_inference",
    #     default=False,
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--data_index",
    #     type=int,
    #     required=True,
    # )
    args = parser.parse_args()
    return args


def visualize_data_only_image(image, coords_2d=None, mano_pose=None):
    n_cols = 2
    n_rows = 2
    fig, axs = plt.subplots(n_cols, n_rows, figsize=(9, 9))
    axs = axs.flatten()
    ax = axs[0]
    ax.set_title("image")
    ax.imshow(image)
    # ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c="red", alpha=0.75)
    plt.tight_layout()
    plt.show()


HandMeta = namedtuple("HandMeta", "scale joints_2d, joints_3d")


def add_ones_column(x):
    additional = torch.ones((x.shape[0], 1))
    return torch.cat([x, additional], dim=1)


def load_data(meta_filepath, image_filepath):
    b = meta_filepath.read_bytes()
    d = pickle.loads(b)
    image = Image.open(image_filepath)
    image = torchvision.transforms.functional.to_tensor(image)
    mano_pose = d["mano_pose"]
    trans = d["trans"]
    pose = np.concatenate([mano_pose, trans])
    pose = torch.from_numpy(pose)
    betas = torch.from_numpy(d["shape"])
    scale = d["z"]
    coords_2d = d["coords_2d"]
    joints_2d = torch.from_numpy(coords_2d)
    joints_2d = add_ones_column(joints_2d)
    coords_3d = d["coords_3d"]
    joints_3d = torch.from_numpy(coords_3d)
    joints_3d = add_ones_column(joints_3d)
    return image, pose, betas, HandMeta(scale, joints_2d, joints_3d), d


######################
# [ok!] ori_img torch.Size([3, 224, 224])
# [ok!] pose    torch.Size([48])
# [ok!] betas   torch.Size([10])
# [ok!] scale   val=0.8026036997235448
######################
# mjm_mask        torch.Size([21, 1])
# mvm_mask        torch.Size([195, 1])

# joints_3d       torch.Size([21, 4])
# joints_2d       torch.Size([21, 3])


if __name__ == "__main__":
    args = parse_args()
    meta_filepath = args.base_path / "datageneration/tmp/meta/00000000.pkl"
    image_filepath = args.base_path / "datageneration/tmp/rgb/00000000.jpg"
    image, pose, betas, meta, data = load_data(meta_filepath, image_filepath)
    center = torch.tensor([112.0, 112])
    print("ori_img: ", image.shape)
    print("pose: ", pose.shape)
    print("betas: ", betas.shape)
    print("center: ", center)
    print(f"scale: {meta.scale}")
    print(f"joints_3d: {meta.joints_3d.shape}")
    print(f"joints_2d: {meta.joints_2d.shape}")
    has_2d_joints = 1
    has_3d_joints = 1
    has_smpl = 1
# visualize_data(image)
# main(data_index=)
