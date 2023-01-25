from pathlib import Path
# import os.path as osp

import torch
import torch.nn.functional as F
from torch import nn, optim
from timm.scheduler import CosineLRScheduler

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
    checkpoint_dir = output_dir / f"checkpoint-{epoch}"
    checkpoint_dir.mkdir(exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model

    torch.save(model_to_save, checkpoint_dir / "model.bin")
    torch.save(model_to_save.state_dict(), checkpoint_dir / "state_dict.bin")
    print(f"Save checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def train(model, epoch, train_loader, train_datasize, optimizer, scheduler, device):
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
        # output = torch.flatten(output)
        # print(f"output: {output.shape}")
        # print(f"data.y: {data.y.shape}")
        # loss = F.nll_loss(output, data.y)
        # print(output.dtype, data.y.dtype)
        gt_y = data.y.view(batch_size, -1).float().contiguous()
        # print(f"gt_y: {gt_y.shape}")
        loss = F.mse_loss(output, gt_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item()) # 損失値の蓄積
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / train_datasize
    print(f'Train Loss: {epoch_loss:.6f}')
    scheduler.step()


def test(model, loader, test_datasize, device):
    model.eval()

    current_loss = 0.0
    # correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            # print(f"output: {output.shape}")
        batch_size = output.shape[0]
        # b = data.y.view(batch_size, -1).float()
        # correct += pred.eq(b).sum().item()
        loss = F.mse_loss(output, data.y.view(batch_size, -1).float().contiguous())
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / test_datasize
    print(f'Validation Loss: {epoch_loss:.6f}')




# def train():
#     model.train()

#     total_loss = correct_nodes = total_nodes = 0
#     for i, data in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.pos, data.batch)
#         loss = F.nll_loss(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
#         total_nodes += data.num_nodes

#         if (i + 1) % 10 == 0:
#             print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
#                   f'Train Acc: {correct_nodes / total_nodes:.4f}')
#             total_loss = correct_nodes = total_nodes = 0


def test(loader):
    model.eval()

    ious, categories = [], []
    y_map = torch.empty(loader.dataset.num_classes, device=device).long()
    for data in loader:
        data = data.to(device)
        outs = model(data.x, data.pos, data.batch)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            iou = jaccard_index(out[:, part].argmax(dim=-1), y_map[y],
                                num_classes=part.size(0), absent_score=1.0)
            ious.append(iou)

        categories.append(data.category)

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)

    mean_iou = scatter(iou, category, reduce='mean')  # Per-category IoU.
    return float(mean_iou.mean())  # Global IoU.


def main(filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_data_for_geometric(filename, device)
    train_datasize = len(train_dataset)
    test_datasize = len(test_dataset)
    print(f"train_datasize={train_datasize} test_datasize={test_datasize}")
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=6, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=6, drop_last=True)

    model = Net().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model = Net(3, train_dataset.num_classes, dim_model=[32, 64, 128, 256, 512], k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, 1000 + 1):
        train(model, epoch, train_loader, train_datasize, optimizer, scheduler, device)
        test(model, test_loader, test_datasize, device)
        if epoch % 5 == 0:
            save_checkpoint(model, epoch)


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    main(filename)
