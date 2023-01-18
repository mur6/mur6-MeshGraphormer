from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch import nn, optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.modeling._mano import MANO

from src.model.pointnet import PointNetfeat, Simple_STN3d
from src.model.pointnet2 import PointNetCls
from src.handinfo.data import load_data
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    point_mesh_face_distance,
)


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


def plane_loss(vert_3d, pca_mean, pca_components):
    # print(vert_3d.shape, pca_mean.shape, pca_components.shape)
    normal_v = torch.cross(pca_components[:, 0], pca_components[:, 1], dim=1).unsqueeze(1)
    # print(f"normal_v: {normal_v.shape}")
    vert_3d = (torch.transpose(vert_3d, 2, 1))
    pca_mean = pca_mean.unsqueeze(1)

    x = (vert_3d - pca_mean) * normal_v
    x = torch.sum(x, dim=2)
    # print(f"sum: {x.shape}")
    # print(f"{x[0]}")
    # print()

    x = torch.pow(x, 2)
    # print(f"square: {x.shape}")
    # print(f"{x[0]}")

    x = torch.sum(x) * 0.05
    # print(x)
    return x


def all_loss(y, y_pred, x, mano_faces):
    #mesh = Meshes(verts=torch.transpose(x, 2, 1), faces=mano_faces)
    loss_1, _ = chamfer_distance(y_pred, y)
    #loss_2 = point_mesh_face_distance(mesh, Pointclouds(torch.transpose(y_pred, 2, 1)))
    return loss_1


def exec_train(train_loader, test_loader, *, model, train_datasize, test_datasize, device, mano_faces, epochs=1000):
    #optimizer = optim.RMSprop(net.parameters(), lr=0.01)
    if False:
        # optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = optim.AdamW(model.parameters(), lr=0.008)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    if True:
        optimizer = optim.AdamW(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0005)
    E = nn.MSELoss()
    # トレーニング
    for epoch in range(epochs):
        losses = []
        current_loss = 0.0
        model.train()
        for i, (x, y, pca_mean, pca_components, normal_v, perimeter) in enumerate(train_loader):
            if device == "cuda":
                x = x.cuda()
                y = y.cuda()
                pca_mean = pca_mean.cuda()
                pca_components = pca_components.cuda()
                normal_v = normal_v.cuda()
                perimeter = perimeter.cuda()
            # print(f"verts: {torch.transpose(x, 2, 1).shape}")
            # print(f"mano_faces: {mano_faces.shape}")
            # print(pca_mean.shape, normal_v.shape)
            optimizer.zero_grad()                   # 勾配情報を0に初期化
            y_pred = model(x)
            # print(f"y_pred: {y_pred.shape}")
            # mean_and_normal_vec = torch.cat((pca_mean, normal_v), dim=1)
            # loss = E(y_pred, y) + plane_loss(y_pred, pca_mean, pca_components)
            loss = all_loss(y, y_pred, x, mano_faces)
            loss.backward()                         # 勾配の計算
            optimizer.step()                        # 勾配の更新
            losses.append(loss.item())              # 損失値の蓄積
            current_loss += loss.item() * y_pred.size(0)

        epoch_loss = current_loss / train_datasize
        print(f'Train Loss: {epoch_loss:.4f}')
        scheduler.step()
        model.eval()
        with torch.no_grad():
            current_loss = 0.0
            for x, y, pca_mean, pca_components, normal_v, perimeter in test_loader:
                if device == "cuda":
                    x = x.cuda()
                    y = y.cuda()
                    pca_mean = pca_mean.cuda()
                    pca_components = pca_components.cuda()
                    normal_v = normal_v.cuda()
                    perimeter = perimeter.cuda()
                y_pred = model(x)
                mean_and_normal_vec = torch.cat((pca_mean, normal_v), dim=1)
                # loss = E(y_pred, y) + plane_loss(y_pred, pca_mean, pca_components)
                loss, _ = chamfer_distance(y_pred, y)
                current_loss += loss.item() * y_pred.size(0)
            epoch_loss = current_loss / test_datasize
            print(f'Validation Loss: {epoch_loss:.4f}')
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, epoch+1)
    with torch.no_grad():
        for x, gt_y in test_loader:
            #print("-------")
            y_pred = model(x)
            y_pred = y_pred.reshape(gt_y.shape)
            print(f"gt: {gt_y[0]} pred: {y_pred[0]}")
            #print(gt_y - y_pred)

    # print(X_train.shape, y_train.shape)
    # plot(X_train, y_train, X_test.data.numpy().T[1], y_pred, losses)


def main(resume_dir, input_filename, device, batch_size):
    train_dataset, test_dataset = load_data(input_filename)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_datasize = len(train_dataset)
    test_datasize = len(test_dataset)
    print(f"train_datasize={train_datasize} test_datasize={test_datasize}")

    if resume_dir:
        if (resume_dir / "model.bin").exists() and \
            (resume_dir / "state_dict.bin").exists():
            model = torch.load(resume_dir / "model.bin")
            state_dict = torch.load(resume_dir / "state_dict.bin")
            model.load_state_dict(state_dict)
        else:
            raise Exception(f"{resume_dir} is not valid directory.")
    else:
        model = PointNetCls()

    mano_model = MANO()

    if device == "cuda":
        model.to(device)
        mano_model = MANO().to(device)
        mano_model.layer = mano_model.layer.cuda()
    mano_faces = mano_model.layer.th_faces
    mano_faces = mano_faces.repeat(batch_size, 1, 1)

    exec_train(
        train_loader, test_loader,
        model=model,
        train_datasize=train_datasize,
        test_datasize=test_datasize,
        device=device,
        mano_faces=mano_faces)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--resume_dir",
        type=Path,
        #required=True,
    )
    parser.add_argument(
        "--input_filename",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.resume_dir, args.input_filename, args.device, args.batch_size)
