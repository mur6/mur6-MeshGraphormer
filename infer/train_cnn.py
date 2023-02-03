from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_cluster import fps, knn_graph
import torch_geometric.transforms as T

from timm.scheduler import CosineLRScheduler

from src.handinfo.data import load_data_for_geometric, get_mano_faces
from src.handinfo.losses import on_circle_loss
from src.model.pointnet import PointNetfeat, Simple_STN3d
from src.model.pointnet2 import PointNetCls


transform = T.Compose([
    # T.RandomJitter(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2),
])
pre_transform = T.NormalizeScale()


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


def exec_train(train_loader, test_loader, *, model, train_datasize, test_datasize, device, epochs=1000):
    #optimizer = optim.RMSprop(net.parameters(), lr=0.01)
    if False:
        # optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = optim.AdamW(model.parameters(), lr=0.009)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    if True:
        optimizer = optim.AdamW(model.parameters(), lr=0.005)
        # optimizer = optim.SGD(model.parameters(), lr=0.005)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0005)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=40,
            cycle_limit=11,
            cycle_decay=0.8,
            lr_min=0.0001,
            warmup_t=20,
            warmup_lr_init=5e-5,
            warmup_prefix=True)
    E = nn.MSELoss()

    for epoch in range(epochs):
        losses = []
        current_loss = 0.0
        model.train()
        for i, (gt_vertices, gt_3d_joints, y, pca_mean, pca_components, normal_v, perimeter) in enumerate(train_loader):
            if device == "cuda":
                gt_vertices = gt_vertices.cuda()
                gt_3d_joints = gt_3d_joints.cuda()
                y = y.cuda()
                pca_mean = pca_mean.cuda()
                pca_components = pca_components.cuda()
                normal_v = normal_v.cuda()
                perimeter = perimeter.cuda()

            optimizer.zero_grad()
            y_pred = model(gt_3d_joints)
            loss = E(pca_mean.float().detach(), y_pred)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            current_loss += loss.item() * y_pred.size(0)

        epoch_loss = current_loss / train_datasize
        print(f'Train Loss: {epoch_loss:.6f}')
        scheduler.step(epoch+1)
        model.eval()
        with torch.no_grad():
            current_loss = 0.0
            for gt_vertices, gt_3d_joints, y, pca_mean, pca_components, normal_v, perimeter in test_loader:
                if device == "cuda":
                    gt_vertices = gt_vertices.cuda()
                    gt_3d_joints = gt_3d_joints.cuda()
                    y = y.cuda()
                    pca_mean = pca_mean.cuda()
                    pca_components = pca_components.cuda()
                    normal_v = normal_v.cuda()
                    perimeter = perimeter.cuda()
                y_pred = model(gt_3d_joints)
                loss = E(pca_mean.float().detach(), y_pred)
                current_loss += loss.item() * y_pred.size(0)
            epoch_loss = current_loss / test_datasize
            print(f'Validation Loss: {epoch_loss:.6f}')
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, epoch+1)



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
        # loss = cyclic_shift_loss(output, gt_y)
        loss = on_circle_loss(output, data)
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
        # loss = cyclic_shift_loss(output, gt_y)
        loss = on_circle_loss(output, data)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"resume_dir: {resume_dir}")
    if resume_dir:
        if (resume_dir / "model.bin").exists() and \
            (resume_dir / "state_dict.bin").exists():
            if torch.cuda.is_available():
                model = torch.load(resume_dir / "model.bin")
                state_dict = torch.load(resume_dir / "state_dict.bin")
            else:
                model = torch.load(resume_dir / "model.bin", map_location=torch.device('cpu'))
                state_dict = torch.load(resume_dir / "state_dict.bin", map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        else:
            raise Exception(f"{resume_dir} is not valid directory.")
    else:
        model = ClassificationNet(
            in_channels=3,
            out_channels=7,
            dim_model=[32, 64, 128, 256, 512],
            ).to(device)
    print(f"model: {model.__class__.__name__}")

    # model = SegmentationNet(
    #     in_channels=3,
    #     out_channels=3,
    #     dim_model=[32, 64, 128, 256, 512],
    # ).to(device)
    model.eval()

    gamma = float(args.gamma)
    print(f"gamma: {gamma}")

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
    if False:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.0001)
    if True:
        optimizer = torch.optim.RAdam(model.parameters(), lr=5e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-05)
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
