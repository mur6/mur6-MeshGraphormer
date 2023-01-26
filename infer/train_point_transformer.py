from pathlib import Path
import argparse

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

from src.handinfo.data import load_data_for_geometric, get_mano_faces
from src.model.transformer import ClassificationNet, SegmentationNet
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance



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
    # T.RandomJitter(0.01),
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


def all_loss(pred_output, gt_y, data, faces):
    # print(out_points.shape)
    pcls = Pointclouds(pred_output.view(-1, 20, 3).contiguous())
    verts = data.x.view(-1, 778, 3).contiguous()
    # loss_1, _ = chamfer_distance(pred_output, y)
    meshes = Meshes(verts=verts, faces=faces)
    loss = point_mesh_face_distance(meshes, pcls)
    return F.mse_loss(pred_output, gt_y) + loss


def cyclic_shift_loss(pred_output, gt_y):
    pred_output = pred_output.view(-1, 20, 3)
    gt_y = gt_y.view(-1, 20, 3)
    # print(f"gt_y: {gt_y.shape}")
    # print(f"pred_output: {pred_output.shape}")
    lis = [torch.roll(pred_output, i, dims=1) for i in range(0, pred_output.shape[1])]
    x = torch.stack(lis, dim=0)
    # print(f"x: {x.shape}")
    tmp_x = (x - gt_y).pow(2).sum(dim=(1, 2, 3))
    final_y_pred = x[torch.argmin(tmp_x)]
    # print(f"final_y_pred: {final_y_pred.shape}")
    return F.mse_loss(final_y_pred, gt_y)


def train(model, device, train_loader, train_datasize, bs_faces, optimizer):
    model.train()
    losses = []
    current_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        # print(f"data.x: {data.x.shape}")
        # print(f"data.pos: {data.pos.shape}")
        optimizer.zero_grad()
        output = model(data.x, data.pos, data.batch)
        # print(f"data.y: {data.y.shape}")
        # print(f"output: {output.shape}")

        batch_size = output.shape[0]
        #print(f"verts: {verts.shape}")
        #print(f"faces: {faces.shape}")

        #.view(batch_size, 1538, 3)
        # print(f"bs_faces: {bs_faces.shape}")
        gt_y = data.y.view(batch_size, -1).float().contiguous()
        # loss = all_loss(output, gt_y, data, bs_faces)
        # loss = F.mse_loss(output, gt_y)
        loss = cyclic_shift_loss(output, gt_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item()) # 損失値の蓄積
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / train_datasize
    print(f'Train Loss: {epoch_loss:.6f}')


def test(model, device, test_loader, test_datasize, bs_faces):
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
        gt_y = data.y.view(batch_size, -1).float().contiguous()
        # loss = all_loss(output, gt_y, data, bs_faces)
        # loss = F.mse_loss(output, gt_y)
        loss = cyclic_shift_loss(output, gt_y)
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / test_datasize
    print(f'Validation Loss: {epoch_loss:.6f}')


def main(resume_dir, input_filename, batch_size, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_data_for_geometric(
        input_filename,
        transform=transform,
        pre_transform=None,
        device=device)
    train_datasize = len(train_dataset)
    test_datasize = len(test_dataset)
    print(f"train_datasize={train_datasize} test_datasize={test_datasize}")

    # batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = ClassificationNet(
        in_channels=3,
        out_channels=20*3,
        dim_model=[32, 64, 128, 256, 512],
        ).to(device)
    # model = SegmentationNet(
    #     in_channels=3,
    #     out_channels=3,
    #     dim_model=[32, 64, 128, 256, 512],
    # ).to(device)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    gamma = float(args.gamma)
    print(f"gamma: {gamma}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=gamma)
    ####### test:
    # for d in train_loader:
    #     print(d.x.shape)
    #     output = model(d.x, d.pos, d.batch)
    #     print(output.shape)
    #     break

    faces = get_mano_faces()
    bs_faces = faces.repeat(batch_size, 1).view(batch_size, 1538, 3)

    for epoch in range(1, 1000 + 1):
        train(model, device, train_loader, train_datasize, bs_faces, optimizer)
        test(model, device, test_loader, test_datasize, bs_faces)
        if epoch % 5 == 0:
            save_checkpoint(model, epoch)
        scheduler.step(epoch)
        print(f"lr: {scheduler.get_last_lr()}")
        # train(model, device, train_loader, optimizer)
        # iou = test(model, device, test_loader)
        # print(f'Epoch: {epoch:03d}, Test IoU: {iou:.4f}')
        # scheduler.step()


def parse_args():
    from decimal import Decimal
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=Decimal, default=Decimal("0.85"))
    parser.add_argument(
        "--resume_dir",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "--input_filename",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.resume_dir, args.input_filename, args.batch_size, args)
