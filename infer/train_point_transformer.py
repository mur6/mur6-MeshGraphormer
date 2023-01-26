from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_cluster import fps, knn_graph

from timm.scheduler import CosineLRScheduler

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
from torch_geometric.utils import scatter

from src.handinfo.data import load_data_for_geometric
from src.model.transformer import ClassificationNet, SegmentationNet



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



transform = T.Compose([
    T.RandomJitter(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2),
])
pre_transform = T.NormalizeScale()

# train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
#                          pre_transform=pre_transform)
# test_dataset = ShapeNet(path, category, split='test',
#                         pre_transform=pre_transform)
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)



def train(model, device, train_loader, train_datasize, optimizer, scheduler):
    model.train()
    losses = []
    current_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        # print(f"data.pos: {data.pos.shape}")
        optimizer.zero_grad()
        output = model(data.x, data.pos, data.batch)
        # print(f"data.y: {data.y.shape}")
        # print(f"output: {output.shape}")
        # print()
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


def test(model, device, test_loader, test_datasize):
    model.eval()

    current_loss = 0.0
    # correct = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.pos, data.batch)
            # print(f"output: {output.shape}")
        batch_size = output.shape[0]
        # b = data.y.view(batch_size, -1).float()
        # correct += pred.eq(b).sum().item()
        loss = F.mse_loss(output, data.y.view(batch_size, -1).float().contiguous())
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / test_datasize
    print(f'Validation Loss: {epoch_loss:.6f}')


def main(filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_data_for_geometric(
        filename,
        transform=transform,
        pre_transform=pre_transform,
        device=device)
    train_datasize = len(train_dataset)
    test_datasize = len(test_dataset)
    print(f"train_datasize={train_datasize} test_datasize={test_datasize}")

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = ClassificationNet(
        in_channels=3,
        out_channels=3,
        dim_model=[32, 64, 128, 256, 512],
        ).to(device)
    # model = SegmentationNet()
    model.eval()

    if False:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if False:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=40,
            cycle_limit=11,
            cycle_decay=0.8,
            lr_min=0.0001,
            warmup_t=20,
            warmup_lr_init=5e-5,
            warmup_prefix=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    ####### test:
    # for d in train_loader:
    #     print(d.x.shape)
    #     output = model(d.x, d.pos, d.batch)
    #     print(output.shape)
    #     break

    for epoch in range(1, 1000 + 1):
        train(model, device, train_loader, train_datasize, optimizer, scheduler)
        test(model, device, test_loader, test_datasize)
        if epoch % 5 == 0:
            save_checkpoint(model, epoch)
        scheduler.step(epoch)
        # train(model, device, train_loader, optimizer)
        # iou = test(model, device, test_loader)
        # print(f'Epoch: {epoch:03d}, Test IoU: {iou:.4f}')
        # scheduler.step()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    main(filename)
