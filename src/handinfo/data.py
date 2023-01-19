import operator
import math

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def _iter(*, dict_list, key_name="vert_3d"):
    for val in dict_list:


        


        yield gt_vertices, vert_3d, pca_mean, pca_components, normal_v, perimeter


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
    gt_3d_joints = val['gt_3d_joints']
    vert_3d = val['vert_3d']
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



if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    print(filename)
    val = _load_data(filename)
    keys = [
        'perimeter',
        'gt_3d_joints',
        'gt_vertices',
        'vert_3d',
        # 'center_points_3d',
        'pca_mean_', 'pca_components_'
    ]

    for k in keys:
        print(k, val[k].shape)

    pca_components = val['pca_components_']
    normal_v = torch.cross(pca_components[:, 0], pca_components[:, 1], dim=1)
    print(normal_v, normal_v.shape)
