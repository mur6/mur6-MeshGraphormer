import operator
import math
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split
# from pytorch3d.structures import Meshes


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
    # vert_3d = torch.transpose(vert_3d, 1, 2)
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


def get_mano_faces(device=None):
    mano_model = MANO()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        mano_model = MANO().to(device)
        mano_model.layer = mano_model.layer.cuda()
    mano_faces = mano_model.layer.th_faces
    return mano_faces


def faces_to_edge_index(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    e = torch.from_numpy(mesh.edges_unique).permute(1, 0)
    # print(e2)
    # print(type(e), e.shape)
    edge_index = torch.cat((e, torch.flip(e, dims=[0])), dim=1)
    return edge_index


class MyGraphDataset(InMemoryDataset):
    def __init__(self, data_list,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
        ):
        super().__init__('.', transform=transform, pre_transform=pre_transform)
        # transform=transform,
        self.data, self.slices = self.collate(data_list)


def load_data_for_geometric(filename, *, transform, pre_transform, device="cuda"):
    val = _load_data(filename)
    perimeter = val['perimeter']
    gt_vertices = val['gt_vertices']
    faces = get_mano_faces(device="cpu")
    print(f"faces: {faces.shape}")
    edge_index = faces_to_edge_index(gt_vertices[0], faces)
    # gt_3d_joints = val['gt_3d_joints']
    pca_mean = val['pca_mean_']
    pca_components = val['pca_components_']
    normal_v = torch.cross(pca_components[:, 0], pca_components[:, 1], dim=1)
    # print("normal_v: ", normal_v.shape, normal_v.dtype)
    vert_3d = val['vert_3d']
    # print(f"vert_3d: {vert_3d.shape}")
    vertices = gt_vertices#torch.transpose(gt_vertices, 1, 2)
    data_list = []

    for i, vertex in enumerate(vertices):
        # print(f"perimeter: {perimeter[i]}")
        # print(f"vertex average: {vertex.abs().mean()}")
        # scale = (1 / vertex.abs().max()) * 0.999999
        # print(f"scale: {scale}")
        # print(f"vertex: {vertex.shape}")
        # print(f"face: {face.shape}")
        # print(f"edge_index: {edge_index.shape} {edge_index.dtype}")
        # print(f"vert_3d[i]: {vert_3d[i].shape}")
        perim = perimeter[i]
        radius = perim / (2.0 * math.pi)
        # print(f"radius: {radius}")
        d = Data(x=vertex, pos=vertex, edge_index=edge_index,
                y=vert_3d[i],
                pca_mean=pca_mean[i],
                normal_v=normal_v[i],
                perimeter=perim,
                radius=radius)
        data_list.append(d)
    # print(vertex.shape, edge_index.shape)
    # print(pca_mean.shape)
    data_train, data_test = train_test_split(data_list)

    train_dataset = MyGraphDataset(
        data_train,
        transform=transform,
        pre_transform=pre_transform)
    test_dataset = MyGraphDataset(
        data_test,
        transform=transform,
        pre_transform=pre_transform)
    #return data_train, data_test
    return train_dataset, test_dataset


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

    transform=None
    pre_transform=None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_data_for_geometric(
        filename,
        transform=transform,
        pre_transform=pre_transform,
        device=device)
    def _iter():
        for data in train_dataset:
            # print(data.normal_v)
            yield data.perimeter[0].item()
    perimeter_list = list(_iter())
    perimeter_list = torch.tensor(perimeter_list)
    perimeter_list = perimeter_list / (2.0 * math.pi)
    print(perimeter_list.mean())
    max = torch.max(perimeter_list)
    min = torch.min(perimeter_list)
    print(f"min={min}, max={max}")
