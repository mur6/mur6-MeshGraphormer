from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


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


class STN3d(nn.Module):
    in_features = 778

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, self.in_features, 1)
        self.conv2 = torch.nn.Conv1d(self.in_features, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3*20)
        # self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.in_features)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        # print("1:", x.shape)
        x = x.view(-1, 1024)
        # print("2:", x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        # else:
        #     x = F.relu(self.fc1(x))
        #     x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # iden = Variable(torch.eye(3, dtype=torch.float32).view(1, 9)).repeat(batch_size, 1)
        # x = x + iden
        x = x.view(-1, 3, 20)
        return x


def exec_train(train_loader, test_loader, *, model, train_datasize, test_datasize, epochs=30):
    #optimizer = optim.RMSprop(net.parameters(), lr=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001)
    E = nn.MSELoss()
    # トレーニング
    for epoch in range(epochs):
        losses = []
        current_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()                   # 勾配情報を0に初期化
            y_pred = model(x)
            # print(y_pred.shape)
            # print(y_pred.reshape(y.shape).shape)
            loss = E(y_pred.reshape(y.shape), y)
            loss.backward()                         # 勾配の計算
            optimizer.step()                        # 勾配の更新
            losses.append(loss.item())              # 損失値の蓄積
            current_loss += loss.item() * y_pred.size(0)

        epoch_loss = current_loss / train_datasize
        print(f'Train Loss: {epoch_loss:.4f}')
        scheduler.step()
        with torch.no_grad():
            current_loss = 0.0
            for x, y in test_loader:
                y_pred = model(x)
                loss = E(y_pred, y)
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


def load_data(filename):
    X, y = [], []
    with np.load(filename, allow_pickle=True) as hand_infos:
        for idx, val in enumerate(hand_infos["arr_0"]):
            perimeter = val['perimeter']
            gt_vertices = val['gt_vertices']
            # print(f"gt_vertices: {gt_vertices.shape}")
            # gt_vertices = torch.from_numpy(gt_vertices)
            # gt_vertices = torch.transpose(gt_vertices, 0, 1)
            betas = torch.from_numpy(val['betas'])
            pose = torch.from_numpy(val['pose'])
            center_points = val['center_points']
            if center_points.shape[0] == 20:
                # print(betas.shape)
                # print(pose.shape)
                # print(center_points)
                X.append(gt_vertices)
                y.append(center_points)
    X = torch.FloatTensor(X)
    X = torch.transpose(X, 1, 2)
    y = torch.FloatTensor(y)
    y = torch.transpose(y, 1, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset, test_dataset


def main(resume_dir, input_filename):
    train_dataset, test_dataset = load_data(input_filename)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
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
        model = STN3d()

    exec_train(
        train_loader, test_loader,
        model=model,
        train_datasize=train_datasize,
        test_datasize=test_datasize)


def parse_args():
    parser = argparse.ArgumentParser()
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
    main(args.resume_dir, args.input_filename)
