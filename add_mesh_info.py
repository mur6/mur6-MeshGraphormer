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


def calc_ring_perimeter(ring_contact_part_mesh):
    v = ring_contact_part_mesh.vertices[ring_contact_part_mesh.faces]
    # メッシュを構成する三角形の重心部分を求める
    center_points = np.mean(v, axis=1)
    # PCAで次元削減及び２次元へ投影
    pca = PCA(n_components=3)
    pca.fit(center_points)
    vert_2d = np.dot(center_points, pca.components_.T[:, :2])
    # ConvexHullで均してから外周を測る
    hull = ConvexHull(vert_2d)
    vertices = hull.vertices.tolist() + [hull.vertices[0]]
    perimeter = np.sum(
        [euclidean(x, y) for x, y in zip(vert_2d[vertices], vert_2d[vertices][1:])]
    )
    return perimeter, center_points


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


def visualize(mesh, ring1, ring2):
    color = [102, 102, 102, 64]
    for facet in mesh.facets:
        # for a_color in mesh.visual.face_colors[facet]:
        # print(k)
        mesh.visual.face_colors[facet] = [color, color]

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    ring_contact_part_mesh = calc_ring_contact_part_mesh(
        hand_mesh=mesh, ring1_point=ring1, ring2_point=ring2
    )
    perimeter, center_points = calc_ring_perimeter(ring_contact_part_mesh)
    scene.add_geometry(create_point_geom(ring1, color="red"))
    scene.add_geometry(create_point_geom(ring2, color="blue"))
    scene.show()


def make_joint2d_and_image(meta_info, annotations):
    img_size = 224
    joints_2d = meta_info.joints_2d
    joints_2d = ((joints_2d + 1) * 0.5) * img_size
    ori_img = annotations["ori_img"]
    return joints_2d, ori_img


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


def make_hand_mesh(mano_model, gt_vertices):
    mano_faces = mano_model.layer.th_faces
    # mesh objects can be created from existing faces and vertex data
    return trimesh.Trimesh(vertices=gt_vertices, faces=mano_faces)


def calc_perimeter_and_center_points(mesh, *, ring1, ring2):
    ring_contact_part_mesh = calc_ring_contact_part_mesh(
        hand_mesh=mesh, ring1_point=ring1, ring2_point=ring2
    )
    perimeter, center_points = calc_ring_perimeter(ring_contact_part_mesh)
    return perimeter, center_points


def get_betas_and_perimeter(dataset, num):
    betas_and_perimeter = []
    for meta_info, annotations in iter_meta_info(itertools.islice(dataset, num)):
        # joints_2d, ori_img = make_joint2d_and_image(meta_info, annotations)
        mano_model = MANO().to("cpu")
        # mano_layer = mano_model.layer
        betas = tuple(meta_info.betas.tolist())
        gt_vertices, gt_vertices_sub, ring_finger_point_func = make_gt_infos(mano_model, meta_info, annotations)
        # print(f"gt_vertices:min{torch.min(gt_vertices)}, max={torch.max(gt_vertices)}")
        # print(f"gt_3d_joints:min{torch.min(gt_3d_joints)}, max={torch.max(gt_3d_joints)}")
        # print("gt_vertices", gt_vertices)
        mesh = make_hand_mesh(mano_model, gt_vertices)
        perimeter, center_points = calc_perimeter_and_center_points(
            mesh,
            ring1=ring_finger_point_func(1),
            ring2=ring_finger_point_func(2),
        )
        perimeter = round(perimeter, 6)
        betas_and_perimeter.append((betas, perimeter))
    betas_and_perimeter = set(betas_and_perimeter)
    return list(betas_and_perimeter)


def main(args, *, train_yaml_file, num):
    dataset = build_hand_dataset(train_yaml_file, args, is_train=True)
    betas_and_perimeter = get_betas_and_perimeter(dataset, num)
    s = json.dumps(list, indent=4)
    args.output_json.write(s)


def calc_ring_contact_part_mesh(*, hand_mesh, ring1_point, ring2_point):
    # カットしたい平面の起点と法線ベクトルを求める
    plane_normal = ring2_point - ring1_point
    plane_origin = (ring1_point + ring2_point) / 2
    # print(f"plane_normal:{plane_normal} plane_origin:{plane_origin}")
    # 上記の平面とメッシュの交わる面を求める
    _, face_index = trimesh.intersections.mesh_plane(
        hand_mesh, plane_normal, plane_origin, return_faces=True
    )
    new_triangles = trimesh.Trimesh(hand_mesh.vertices, hand_mesh.faces[face_index])
    # 起点と最も近い面(三角形)を求める
    center_points = np.average(new_triangles.vertices[new_triangles.faces], axis=1)
    distances = numpy.linalg.norm(
        center_points - np.expand_dims(plane_origin, 0), axis=1
    )
    triangle_id = np.argmin(distances)
    # print(f"closest triangle_id: {triangle_id}")
    # 上記の三角形を含む、連なったグループを求める
    closest_face_index = None
    for face_index in trimesh.graph.connected_components(new_triangles.face_adjacency):
        if triangle_id in face_index:
            closest_face_index = face_index

    new_face_index = new_triangles.faces[closest_face_index]
    ring_contact_part = trimesh.Trimesh(new_triangles.vertices, new_face_index)
    return ring_contact_part


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
        "--output_json",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args, train_yaml_file=args.train_yaml, num=args.num)
