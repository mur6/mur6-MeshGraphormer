import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def load_data(filename):
    X, y = [], []
    with np.load(filename, allow_pickle=True) as hand_infos:
        for idx, val in enumerate(hand_infos["arr_0"]):
            perimeter = val['perimeter']
            gt_vertices = val['gt_vertices']
            # print(f"gt_vertices: {gt_vertices.shape}")
            # gt_vertices = torch.from_numpy(gt_vertices)
            # gt_vertices = torch.transpose(gt_vertices, 0, 1)
            betas = torch.from_numpy(val['betas'])
            pose = torch.from_numpy(val['pose'])
            center_points = val['center_points_3d']
            print(center_points.shape)
            if center_points.shape[0] == 19:
                # print(betas.shape)
                # print(pose.shape)
                # print(center_points)
                X.append(gt_vertices)
                y.append(center_points)
    X = torch.FloatTensor(X)
    X = torch.transpose(X, 1, 2)
    y = torch.FloatTensor(y)
    y = torch.transpose(y, 1, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset, test_dataset
