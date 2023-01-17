import argparse
import itertools
import os.path
import json
from collections import namedtuple
from pathlib import Path

import trimesh
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import torch
from PIL import Image

import src.modeling.data.config as cfg
from manopth.manolayer import ManoLayer
from src.datasets.hand_mesh_tsv import HandMeshTSVDataset, HandMeshTSVYamlDataset
from src.modeling._mano import MANO, Mesh

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

from src.handinfo.ring_calc import calc_perimeter_and_center_points


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


def adjust_vertices(gt_vertices, gt_3d_joints):
    gt_vertices = gt_vertices / 1000.0
    gt_3d_joints = gt_3d_joints / 1000.0
    # orig_3d_joints = gt_3d_joints

    mesh_sampler = Mesh(device=torch.device("cpu"))
    gt_vertices_sub = mesh_sampler.downsample(gt_vertices)

    batch_size = 1

    gt_3d_root = gt_3d_joints[:, cfg.J_NAME.index("Wrist"), :]
    gt_vertices = gt_vertices - gt_3d_root[:, None, :]
    gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
    gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
    # gt_3d_joints_with_tag = torch.ones((batch_size, gt_3d_joints.shape[1], 4))
    # gt_3d_joints_with_tag[:, :, :3] = gt_3d_joints
    return gt_vertices.squeeze(0), gt_vertices_sub.squeeze(0), gt_3d_joints.squeeze(0)


def create_point_geom(ring_point, color):
    geom = trimesh.creation.icosphere(radius=0.0008)
    if color == "red":
        color = [202, 2, 2, 255]
    else:
        color = [0, 0, 200, 255]
    geom.visual.face_colors = color
    # print(ring3, ring4)
    # geom.apply_transform(trimesh.transformations.random_rotation_matrix())
    # geom.center = [1, 1, 1]
    geom.apply_translation(ring_point)
    return geom


MANO_JOINTS_NAME = (
    'Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4',
    'Index_1', 'Index_2', 'Index_3', 'Index_4',
    'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4',
    'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4',
    'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4',
)

def make_gt_infos(mano_model, meta_info, annotations):
    pose = meta_info.pose.unsqueeze(0)
    betas = meta_info.betas.unsqueeze(0)
    gt_vertices, gt_3d_joints = mano_model.layer(pose, betas)
    gt_vertices, gt_vertices_sub, gt_3d_joints = adjust_vertices(gt_vertices, gt_3d_joints)

    # print(f"gt_vertices: {gt_vertices.shape}")
    # print(f"gt_vertices_sub: {gt_vertices_sub.shape}")
    # print(f"joints: {joints.shape}")

    def ring_finger_point_func(num):
        ring_point = MANO_JOINTS_NAME.index(f'Ring_{num}')
        return gt_3d_joints[ring_point]

    return gt_vertices, gt_vertices_sub, ring_finger_point_func


def visualize(mesh, ring1, ring2):
    color = [102, 102, 102, 64]
    for facet in mesh.facets:
        # for a_color in mesh.visual.face_colors[facet]:
        # print(k)
        mesh.visual.face_colors[facet] = [color, color]

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(create_point_geom(ring1, color="red"))
    scene.add_geometry(create_point_geom(ring2, color="blue"))
    scene.show()


def make_joint2d_and_image(meta_info, annotations):
    img_size = 224
    joints_2d = meta_info.joints_2d
    joints_2d = ((joints_2d + 1) * 0.5) * img_size
    ori_img = annotations["ori_img"]
    return joints_2d, ori_img


def make_hand_mesh(mano_model, gt_vertices):
    mano_faces = mano_model.layer.th_faces
    # mesh objects can be created from existing faces and vertex data
    return trimesh.Trimesh(vertices=gt_vertices, faces=mano_faces)


OutputInfo = namedtuple("OutputInfo", "betas, gt_vertices, perimeter, center_points")


def get_output_data(dataset, num):
    output_list = []
    count = 0
    for meta_info, annotations in iter_meta_info(itertools.islice(dataset, num)):
        # joints_2d, ori_img = make_joint2d_and_image(meta_info, annotations)
        mano_model = MANO().to("cpu")
        gt_vertices, gt_vertices_sub, ring_finger_point_func = make_gt_infos(mano_model, meta_info, annotations)
        # print(f"gt_vertices:min{torch.min(gt_vertices)}, max={torch.max(gt_vertices)}")
        # print(f"gt_3d_joints:min{torch.min(gt_3d_joints)}, max={torch.max(gt_3d_joints)}")
        # print("gt_vertices", gt_vertices)
        mesh = make_hand_mesh(mano_model, gt_vertices)
        r = calc_perimeter_and_center_points(
            mesh,
            ring1=ring_finger_point_func(1),
            ring2=ring_finger_point_func(2),
            round_perimeter=True
        )
        output_list.append(
            dict(
                betas=meta_info.betas.numpy(),
                pose=meta_info.pose.numpy(),
                gt_vertices=gt_vertices.numpy(),
                perimeter=r.perimeter,
                vert_3d=r.vert_3d,
                center_points=r.center_points,
                center_points_3d=r.center_points_3d,
            )
        )
        print(f"count: {count}")
        count += 1
    # betas_and_perimeter = set(betas_and_perimeter)
    return output_list


def main(args, *, train_yaml_file, num):
    dataset = build_hand_dataset(train_yaml_file, args, is_train=True)
    outputs = get_output_data(dataset, num)
    #s = json.dumps(betas_and_perimeter, indent=4)
    #args.output_json.write_text(s)
    np.savez(args.output, outputs)



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
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args, train_yaml_file=args.train_yaml, num=args.num)
