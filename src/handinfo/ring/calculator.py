import dataclasses
from dataclasses import dataclass
from collections import namedtuple
from pathlib import Path
import math

import trimesh

# import matplotlib.pyplot as plt
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


def _calc_ring_contact_part_mesh(*, hand_mesh, ring1_point, ring2_point):
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
    distances = numpy.linalg.norm(center_points - np.expand_dims(plane_origin, 0), axis=1)
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


@dataclass(frozen=True)
class RingPointsInfo:
    perimeter: np.ndarray
    radius: np.ndarray
    # p = 2πr
    vert_2d: np.ndarray
    vert_3d: np.ndarray
    center_points: np.ndarray
    center_points_3d: np.ndarray
    pca_mean_: np.ndarray
    pca_components_: np.ndarray


def _round_and_conv_np(perimeter):
    perimeter = round(perimeter, 6)
    perimeter = np.array(perimeter)
    return perimeter


def _calc_ring_perimeter(ring_contact_part_mesh):
    v = ring_contact_part_mesh.vertices[ring_contact_part_mesh.faces]
    # メッシュを構成する三角形の重心部分を求める
    center_points = np.mean(v, axis=1)
    # PCAで次元削減及び２次元へ投影
    pca = PCA(n_components=2).fit(center_points)
    # vert_2d = np.dot(center_points, pca.components_.T[:, :2])
    vert_2d = pca.transform(center_points)
    vert_3d = pca.inverse_transform(vert_2d)
    # print(f"pca.mean: {pca.mean_}")
    # normal_v = np.cross(pca.components_[0], pca.components_[1])
    # print(f"normal_v: {normal_v}")
    # a = (vert_3d - pca.mean_) * normal_v
    # print(a.sum(axis=1))

    # ConvexHullで均してから外周を測る
    hull = ConvexHull(vert_2d)
    vertices = hull.vertices.tolist() + [hull.vertices[0]]

    # fix vert_3d
    if vert_3d.shape[0] > 20:
        vert_3d = vert_3d[:20, :]
    # print(vert_3d.shape)
    if vert_3d.shape != (20, 3):
        return None

    perimeter = np.sum([euclidean(x, y) for x, y in zip(vert_2d[vertices], vert_2d[vertices][1:])])
    perimeter = _round_and_conv_np(perimeter)
    radius = perimeter / (2.0 * math.pi)
    center_points_3d = pca.inverse_transform(vert_2d[vertices])
    return RingPointsInfo(
        perimeter=perimeter,
        radius=radius,
        vert_2d=vert_2d,
        vert_3d=vert_3d,
        center_points=np.array(center_points),
        center_points_3d=center_points_3d,
        pca_mean_=pca.mean_,
        pca_components_=pca.components_,
    )


def _calc_perimeter_and_center_points(*, hand_meshes, ring1s, ring2s):
    for mesh, ring1_point, ring2_point in zip(hand_meshes, ring1s, ring2s):
        ring_contact_part_mesh = _calc_ring_contact_part_mesh(
            hand_mesh=mesh, ring1_point=ring1_point, ring2_point=ring2_point
        )
        ring_points_info = _calc_ring_perimeter(ring_contact_part_mesh)
        if ring_points_info is not None:
            d = dataclasses.asdict(ring_points_info)
            yield d


def calc_perimeter_and_center_points(**kwargs):
    return list(_calc_perimeter_and_center_points(**kwargs))
