# import os.path as osp
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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

def _backup_on_circle_loss(x, pca_mean, normal_v):
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

def on_circle_loss(verts_3d, pred_pca_mean, pred_normal_v):
    radius = torch.FloatTensor([0.0095])
    # radius torch.full((2, 3), 3.141592)
    # batch_size = pred_output.shape[0]
    # # pred_output
    # print(f"pred_output: {pred_output.shape}")
    # # (x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2 = r^2
    # pca_mean = data.pca_mean.view(batch_size, -1)
    # normal_v = data.normal_v.view(batch_size, -1)
    print(f"verts_3d: {verts_3d.shape}")

    d =  (pred_normal_v * pred_pca_mean).sum(dim=-1) #  a*x_0 + b*y_0 + c*z_0
    pred_normal_v  = pred_normal_v.unsqueeze(-2)
    loss_2 = (verts_3d * pred_normal_v).sum(dim=(1, 2)) - d

    print(f"loss_2: {loss_2.shape}")
    # print(loss_1)
    pred_pca_mean = pred_pca_mean.unsqueeze(-2)
    print(f"pred_pca_mean: {pred_pca_mean.shape}")
    # print(f"normal_v: {normal_v.shape}")
    #print(x[:, :3].shape)
    loss_1 =  (verts_3d - pred_pca_mean).pow(2).sum(dim=(1, 2)) - radius.pow(2)
    loss_1 = loss_1 * 0.1
    # print(f"loss_1: {loss_1.shape}")
    print(loss_1)
    print(loss_2)
    loss = torch.cat((loss_1.pow(2), loss_2.pow(2)))
    return loss.sum()
#+ F.mse_loss()


# pred_output: torch.Size([32, 6])
# pca_mean: torch.Size([32, 3])
# normal_v: torch.Size([32, 3])
def new_loss_func_main():
    verts_3d = torch.randn(4, 20, 3)#: torch.Size([32, 20, 3])
    pred_output = torch.randn(4, 6)
    pca_mean = torch.randn(4, 3)
    normal_v = torch.randn(4, 3)
    loss = on_circle_loss(verts_3d, pca_mean, normal_v)
    print(loss)


def calc_circle(a, b, radius):
    theta = torch.linspace(-math.pi, math.pi, steps=100)
    theta = theta.expand(3, -1).t()
    # print(theta.shape)
    z = a.unsqueeze(0) * torch.cos(theta) + b.unsqueeze(0) * torch.sin(theta)
    return z * radius


def main():
    # pred: mean: tensor([-0.0285,  0.0415, -0.0629])
    # gt: mean: tensor([-0.0247,  0.0446, -0.0663], dtype=torch.float64)
    # pred: radius: tensor([0.0091])
    # gt: radius: tensor([0.0082], dtype=torch.float64)
    # pred: normal_v: tensor([ 1.0154e-03, -4.3981e-05, -2.5771e-03])
    # gt:normal_v: tensor([ 0.4021, -0.2091, -0.8914], dtype=torch.float64)
    mean = torch.tensor([-0.0247,  0.0446, -0.0663])
    print(mean)

    input = torch.tensor([[12., -51, 4], [0, 1, 0], [0, 0, 1]]).t()
    # print(a)
    q, r = torch.linalg.qr(input)
    #print(r)
    print(q[:, 0] * -52.545)
    a = q[:, 1]
    b = q[:, 2]
    z = calc_circle(a, b, 0.09)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    print(z.shape)
    for xs, ys, zs in z:
        ax.scatter(xs, ys, zs, marker="o")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    # print(a, torch.linalg.vector_norm(a))
    # print(b, torch.linalg.vector_norm(b))
    # print(q[:, 0] @ q[:, 1])
    # print(q[:, 0] @ q[:, 2])
    # print(q[:, 1] @ q[:, 2])
main()
