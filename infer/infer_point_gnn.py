import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

from src.model.geometric import GlobalSAModule, SAModule, Net
from src.handinfo.data import load_data_for_geometric
# from src. pointnet2_classification import GlobalSAModule, SAModule


def train(epoch):
    model.train()

    for data in train_loader:
        # print(type(data))
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(f"output: {output.shape}")
        batch_size = output.shape[0]
        # print(f"data.y: {data.y.shape}")
        # print(output)
        # output = torch.flatten(output)
        # print(output)
        # loss = F.nll_loss(output, data.y)
        # print(output.dtype, data.y.dtype)
        loss = F.mse_loss(output, data.y.view(batch_size, -1).float().contiguous())
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            # print(f"output: {output.shape}")
            pred = output#output.max(1)[1]
        batch_size = pred.shape[0]
        b = data.y.view(batch_size, -1).float()
        correct += pred.eq(b).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    print(filename)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                    'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_data_for_geometric(filename, device)
    # train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    # test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    # print(train_dataset.data)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        # train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')
