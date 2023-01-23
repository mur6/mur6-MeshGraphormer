from pathlib import Path
# import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

from src.model.geometric import GlobalSAModule, SAModule, Net
from src.handinfo.data import load_data_for_geometric
# from src. pointnet2_classification import GlobalSAModule, SAModule


def save_checkpoint(model, epoch, iteration=None):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoint-{epoch}" # -{iteration}
    checkpoint_dir.mkdir(exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model

    torch.save(model_to_save, checkpoint_dir / "model.bin")
    torch.save(model_to_save.state_dict(), checkpoint_dir/ "state_dict.bin")
    print(f"Save checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def train(train_loader, train_datasize):
    model.train()
    losses = []
    current_loss = 0.0
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
        losses.append(loss.item()) # 損失値の蓄積
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / train_datasize
    print(f'Train Loss: {epoch_loss:.6f}')


def test(loader, test_datasize):
    model.eval()

    current_loss = 0.0
    # correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            # print(f"output: {output.shape}")
        batch_size = pred.shape[0]
        # b = data.y.view(batch_size, -1).float()
        # correct += pred.eq(b).sum().item()
        loss = F.mse_loss(output, data.y.view(batch_size, -1).float().contiguous())
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / test_datasize
    print(f'Validation Loss: {epoch_loss:.6f}')

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    print(filename)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
    #                 'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_data_for_geometric(filename, device)
    train_datasize = len(train_dataset)
    test_datasize = len(test_dataset)
    print(f"train_datasize={train_datasize} test_datasize={test_datasize}")
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

    for epoch in range(1, 500 + 1):
        train(train_loader, train_datasize)
        test(test_loader, test_datasize)
        # test_acc = test(test_loader)
        # print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')
        if epoch % 5 == 0:
            save_checkpoint(model, epoch)
