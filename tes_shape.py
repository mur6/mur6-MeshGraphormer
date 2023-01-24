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


x = torch.arange(8*3)
#x = torch.reshape(x, (8, 3))
#print(x)
x = x.view(8, -1)
print(x)
