import operator

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def _iter(*, dict_list):
    for val in dict_list:
        perimeter = val['perimeter']
        gt_vertices = val['gt_vertices']
        # print(f"gt_vertices: {gt_vertices.shape}")
        # gt_vertices = torch.from_numpy(gt_vertices)
        # gt_vertices = torch.transpose(gt_vertices, 0, 1)
        betas = torch.from_numpy(val['betas'])
        pose = torch.from_numpy(val['pose'])
        center_points_3d = np.zeros((21, 3), dtype=np.float64)
        x = val['center_points_3d']
        center_points_3d[:x.shape[0], :] = x
        # if center_points.shape[0] == 19:
        #     # print(betas.shape)
        #     # print(pose.shape)
        #     # print(center_points)
        #     X.append(gt_vertices)
        #     y.append(center_points_3d)
        yield gt_vertices, center_points_3d


def load_data(filename):
    # X, y = [], []
    values = None
    with np.load(filename, allow_pickle=True) as hand_infos:
        values = list(_iter(dict_list=hand_infos["arr_0"]))

    X = list(map(operator.itemgetter(0), values))
    X = torch.FloatTensor(X)
    X = torch.transpose(X, 1, 2)
    # print(X.shape)

    y = list(map(operator.itemgetter(1), values))
    y = torch.FloatTensor(y)
    y = torch.transpose(y, 1, 2)
    # print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset, test_dataset
