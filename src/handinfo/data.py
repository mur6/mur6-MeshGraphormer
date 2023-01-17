import operator

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def _iter(*, dict_list, key_name="vert_3d"):
    for val in dict_list:
        # perimeter = val['perimeter']
        gt_vertices = val['gt_vertices']
        # betas = torch.from_numpy(val['betas'])
        # pose = torch.from_numpy(val['pose'])
        vert_3d = val['vert_3d']
        pca_mean = val['pca_mean_']
        pca_components = val['pca_components_']
        if vert_3d.shape != (20, 3):
            # print(y.shape)
            vert_3d = vert_3d[:20, :]
        assert vert_3d.shape == (20, 3)
        yield gt_vertices, vert_3d, pca_mean, pca_components




def load_data(filename):
    # X, y = [], []
    values = None
    with np.load(filename, allow_pickle=True) as hand_infos:
        values = list(_iter(dict_list=hand_infos["arr_0"]))

    def conv_float_tensor(index):
        data = list(map(operator.itemgetter(index), values))
        return torch.FloatTensor(data)

    X = conv_float_tensor(0)
    X = torch.transpose(X, 1, 2)
    # print(X.shape)

    X = conv_float_tensor(1)
    y = torch.transpose(y, 1, 2)
    # print(y.shape)

    pca_mean = conv_float_tensor(2)
    pca_components = conv_float_tensor(3)

    (
        X_train, X_test, y_train, y_test,
        pca_mean_train, pca_mean_test,
        pca_components_train, pca_components_test
    ) = train_test_split(X, y, pca_mean, pca_components)
    train_dataset = TensorDataset(X_train, y_train, pca_mean_train, pca_components_train)
    test_dataset = TensorDataset(X_test, y_test, pca_mean_test, pca_components_test)
    return train_dataset, test_dataset
