import argparse
import itertools
import os.path
from collections import namedtuple
from pathlib import Path

# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
# from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import src.modeling.data.config as cfg
from src.modeling._mano import MANO, Mesh
from manopth.manolayer import ManoLayer
from src.datasets.hand_mesh_tsv import HandMeshTSVDataset, HandMeshTSVYamlDataset


def show_3d_plot_just_one(axs, points3d, alpha=None, color=None, with_index=False):
    X, Y, Z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    axs.scatter(X, Y, Z, alpha=alpha, color=color)
    if with_index:
        for i in range(len(X)):
            axs.text(X[i], Y[i], Z[i], str(i), color='blue')


def visualize_data_3d(gt_vertices_sub, gt_3d_joints):
    verts = gt_vertices_sub[0]
    joints = gt_3d_joints[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('Z')
    show_3d_plot_just_one(ax, verts, alpha=0.1)
    show_3d_plot_just_one(ax, joints, color="red", with_index=True)
    plt.show()


def build_hand_dataset(yaml_file, args, is_train=True, scale_factor=1):
    if not os.path.isfile(yaml_file):
        yaml_file = os.path.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert os.path.isfile(yaml_file)
    return HandMeshTSVYamlDataset(args, yaml_file, is_train, False, scale_factor)


MetaInfo = namedtuple("MetaInfo", "pose,betas,joints_2d,joints_3d")


def iter_meta_info(dataset_partial):
    for img_key, transfromed_img, meta_data in dataset_partial:
        pose = meta_data["pose"]
        assert pose.shape == (48,)
        # mano_pose, trans = pose[:45], pose[45:]
        # assert mano_pose.shape == (45,)
        # assert trans.shape == (3,)
        betas = meta_data["betas"]
        assert betas.shape == (10,)
        joints_2d = meta_data["joints_2d"][:, 0:2]
        assert joints_2d.shape == (21, 2)
        joints_3d = meta_data["joints_3d"][:, 0:3]
        assert joints_3d.shape == (21, 3)
        # print(mano_pose.shape, trans.shape, betas.shape, joints_2d.shape, joints_3d.shape)
        yield MetaInfo(pose, betas, joints_2d, joints_3d), meta_data


def visualize_data(ori_img, joints_2d):
    # n_cols = 2
    # n_rows = 2
    # fig, axs = plt.subplots(n_cols, n_rows, figsize=(9, 9))
    # axs = axs.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("ori_img & joints_2d")
    ax.imshow(ori_img)
    ax.scatter(joints_2d[:, 0], joints_2d[:, 1], c="red", alpha=0.75)
    plt.tight_layout()
    plt.show()

def main(args, *, train_yaml_file, num):
    dataset = build_hand_dataset(train_yaml_file, args, is_train=True)
    meta_info, annotations = list(iter_meta_info(itertools.islice(dataset, num)))[0]
    print(f"joints_2d: {meta_info.joints_2d}")
    img_size = 224
    joints_2d = meta_info.joints_2d
    joints_2d = ((joints_2d + 1) * 0.5) * img_size
    ori_img = annotations["ori_img"]
    #visualize_data(ori_img.numpy().transpose(1,2,0), joints_2d)
    mano_model = MANO().to("cpu")
    # mano_model.layer = mano_model.layer.cuda()
    mano_layer = mano_model.layer
    mesh_sampler = Mesh(device=torch.device('cpu'))

    pose = meta_info.pose.unsqueeze(0)
    betas = meta_info.betas.unsqueeze(0)
    gt_vertices, gt_3d_joints = mano_model.layer(pose, betas)
    gt_vertices = gt_vertices / 1000.0
    gt_3d_joints = gt_3d_joints / 1000.0
    gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
    # print(gt_vertices.shape, gt_3d_joints.shape)
    # normalize gt based on hand's wrist
    batch_size = 1

    gt_3d_root = gt_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
    gt_vertices = gt_vertices - gt_3d_root[:, None, :]
    gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
    gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
    gt_3d_joints_with_tag = torch.ones((batch_size, gt_3d_joints.shape[1],4))
    gt_3d_joints_with_tag[:,:,:3] = gt_3d_joints

    print(f"gt_3d_joints:{gt_3d_joints.shape} gt_vertices:{gt_vertices.shape}")
    print(f"gt_vertices:min{torch.min(gt_vertices)}, max={torch.max(gt_vertices)}")
    print(f"gt_3d_joints:min{torch.min(gt_3d_joints)}, max={torch.max(gt_3d_joints)}")
    print("gt_3d_joints", gt_3d_joints)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # #verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][batch_idx]
    # #if mano_faces is None:
    visualize_data_3d(gt_vertices, gt_3d_joints)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiscale_inference",
        default=False,
        action="store_true",
    )
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args, train_yaml_file=args.train_yaml, num=args.num)
