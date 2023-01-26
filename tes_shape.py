import os.path as osp

import torch
import torch.nn.functional as F
# from pointnet2_classification import GlobalSAModule, SAModule
# from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, knn_interpolate
from torch_geometric.utils import scatter
from torch_geometric.data import Data, InMemoryDataset

from pathlib import Path

def main_1():
    category = 'Airplane'
    path = str(Path('./data/ShapeNet/'))
    transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()
    # train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
    #                          pre_transform=pre_transform)
    test_dataset = ShapeNet(path, category, split='test',
                            pre_transform=pre_transform)

def main_2():
    mlp = MLP([3, 778, 778, 128])
    mlp.eval()
    input = torch.randn(2, 778, 3)
    output = mlp(input)
    print(output.shape)


# x = torch.arange(8*3)
# #x = torch.reshape(x, (8, 3))
# #print(x)
# x = x.view(8, -1)
# print(x)


# x = torch.tensor([1, 2, 3])
# x = x.repeat(4).view(4, 3)
# print(x)

def on_circle_loss(x, pca_mean, normal_v):
    radius = torch.FloatTensor([0.0095])
    # radius torch.full((2, 3), 3.141592)
    # batch_size = pred_output.shape[0]
    # # pred_output
    # print(f"pred_output: {pred_output.shape}")
    # # (x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2 = r^2
    # pca_mean = data.pca_mean.view(batch_size, -1)
    # normal_v = data.normal_v.view(batch_size, -1)
    # print(f"pca_mean: {pca_mean.shape}")
    # print(f"normal_v: {normal_v.shape}")
    #print(x[:, :3].shape)
    loss_1 =  (x[:, :3] - pca_mean).pow(2).sum(dim=-1) - radius.pow(2)
    print(f"loss_1: {loss_1.shape}")

    d =  (normal_v * pca_mean).sum(dim=-1) #  a*x_0 + b*y_0 + c*z_0
    loss_2 = (normal_v * x[:, 3:]).sum(dim=-1) - d

    print(f"loss_2: {loss_2.shape}")
    print(loss_1)
    print(loss_2)
    loss = torch.cat((loss_1.pow(2), loss_2.pow(2)))
    return loss.sum()


# pred_output: torch.Size([32, 6])
# pca_mean: torch.Size([32, 3])
# normal_v: torch.Size([32, 3])
pred_output = torch.randn(4, 6)
pca_mean = torch.randn(4, 3)
normal_v = torch.randn(4, 3)
loss = on_circle_loss(pred_output, pca_mean, normal_v)
print(loss)
