import operator
import math

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def _load_data(filename):
    converted = {}
    with np.load(filename, allow_pickle=True) as val:
        keys = (
            'perimeter',
            'gt_3d_joints',
            'gt_vertices',
            'pca_mean_',
            'pca_components_'
        )
        for k in keys:
            converted[k] = torch.from_numpy(val[k])
        def _conv_vert_3d():
            for vert_3d in val['vert_3d']:
                if vert_3d.shape[0] > 20:
                    vert_3d = vert_3d[:20, :]
                assert vert_3d.shape == (20, 3)
                yield vert_3d
        values = list(_conv_vert_3d())
        converted['vert_3d'] = torch.from_numpy(np.array(values))
    return converted


def load_data(filename):
    val = _load_data(filename)
    perimeter = val['perimeter']
    gt_vertices = val['gt_vertices']
    gt_vertices = torch.transpose(gt_vertices, 1, 2)
    gt_3d_joints = val['gt_3d_joints']
    gt_3d_joints = torch.transpose(gt_3d_joints, 1, 2)
    vert_3d = val['vert_3d']
    vert_3d = torch.transpose(vert_3d, 1, 2)
    pca_mean = val['pca_mean_']
    pca_components = val['pca_components_']

    normal_v = torch.cross(pca_components[:, 0], pca_components[:, 1], dim=1)
    (
        gt_vertices_train, gt_vertices_test,
        gt_3d_joints_train, gt_3d_joints_test,
        vert_3d_train, vert_3d_test,
        pca_mean_train, pca_mean_test,
        pca_components_train, pca_components_test,
        normal_v_train, normal_v_test,
        perimeter_train, perimeter_test,
    ) = train_test_split(gt_vertices, gt_3d_joints, vert_3d, pca_mean, pca_components, normal_v, perimeter)
    train_dataset = TensorDataset(
        gt_vertices_train,
        gt_3d_joints_train,
        vert_3d_train,
        pca_mean_train,
        pca_components_train,
        normal_v_train,
        perimeter_train,
    )
    test_dataset = TensorDataset(
        gt_vertices_test,
        gt_3d_joints_test,
        vert_3d_test,
        pca_mean_test,
        pca_components_test,
        normal_v_test,
        perimeter_test,
    )
    return train_dataset, test_dataset


import trimesh
from src.modeling._mano import MANO


def get_mano_faces(device):
    mano_model = MANO()
    if device == "cuda":
        mano_model = MANO().to(device)
        mano_model.layer = mano_model.layer.cuda()
    mano_faces = mano_model.layer.th_faces
    return mano_faces

from torch_geometric.data import Data, InMemoryDataset


def faces_to_edge_index(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    e = torch.from_numpy(mesh.edges_unique).permute(1, 0)
    # print(e2)
    # print(type(e), e.shape)
    edge_index = torch.cat((e, torch.flip(e, dims=[0])), dim=1)
    return edge_index


def load_data_for_geometric(filename, device="cuda"):
    val = _load_data(filename)
    perimeter = val['perimeter']
    gt_vertices = val['gt_vertices']
    edge_index = faces_to_edge_index(gt_vertices[0], get_mano_faces(device))
    # gt_3d_joints = val['gt_3d_joints']
    pca_mean = val['pca_mean_']
    # pca_components = val['pca_components_']
    vertices = gt_vertices#torch.transpose(gt_vertices, 1, 2)
    data_list = []
    for i, vertex in enumerate(vertices):
        # print(f"vertex: {vertex.shape}")
        # print(f"edge_index: {edge_index.shape} {edge_index.dtype}")
        d = Data(x=None, pos=vertex, edge_index=edge_index, y=pca_mean[i])
        data_list.append(d)
    # print(vertex.shape, edge_index.shape)
    # print(pca_mean.shape)
    data_train, data_test = train_test_split(data_list)
    # train_dataset = InMemoryDataset(".")
    # train_dataset.data = data_train
    # test_dataset = InMemoryDataset(".")
    # test_dataset.data = data_test
    # return train_dataset, test_dataset
    return data_train, data_test


def test1(filename):
    val = _load_data(filename)
    keys = [
        'perimeter',
        'gt_3d_joints',
        'gt_vertices',
        'vert_3d',
        # 'center_points_3d',
        'pca_mean_',
        'pca_components_'
    ]

    for k in keys:
        v = val[k]
        print(k, v.shape, v.dtype)

    pca_components = val['pca_components_']
    normal_v = torch.cross(pca_components[:, 0], pca_components[:, 1], dim=1)
    print("normal_v: ", normal_v.shape, normal_v.dtype)


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    print(filename)
    load_data_for_geometric(filename, device="cpu")
