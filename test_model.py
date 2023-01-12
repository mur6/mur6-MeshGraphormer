from pathlib import Path
import json

import torch
import torch.nn as nn
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# from models import SpatialDescriptor, StructuralDescriptor, MeshConvolution
from torch.utils.data import TensorDataset, DataLoader

class SpatialDescriptor(nn.Module):

    def __init__(self):
        super(SpatialDescriptor, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, centers):
        return self.spatial_mlp(centers)

# m = SpatialDescriptor()
# input = torch.randn(1, 3, 10)
# # x = torch.FloatTensor((1, 3, 10))
# output = m(input)
# print(output.shape)

class ModelNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 10
        hidden_dim1 = 128
        hidden_dim2 = 512
        hidden_dim3 = 64
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim3, 1),
        )

    def forward(self, input):
        return self.head(input)


class Regression(nn.Module):
    # コンストラクタ(インスタンス生成時の初期化)
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 512)
        self.linear2 = nn.Linear(512, 16)
        self.linear3 = nn.Linear(16, 1)

    # メソッド(ネットワークをシーケンシャルに定義)
    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x



def test(model, x):
    y_pred = model(x).data.numpy().T[0]
    return y_pred

# グラフ描画関数
def plot(x, y, x_new, y_pred, losses):
    # ここからグラフ描画-------------------------------------------------
    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
 
    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
 
    # グラフの上下左右に目盛線を付ける。
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2 = fig.add_subplot(122)
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
 
    # 軸のラベルを設定する。
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('E')
 
    # スケール設定
    ax1.set_xlim(-5, 15)
    ax1.set_ylim(-2, 2)
    ax2.set_xlim(0, 5000)
    ax2.set_ylim(0.001, 100)
    ax2.set_yscale('log')
 
    # データプロット
    #ax1.scatter(x, y, label='dataset')
    #ax1.plot(x_new, y_pred, color='red', label='PyTorch regression', marker="o", markersize=3)
    ax2.plot(np.arange(0, len(losses), 1), losses)
    ax2.text(600, 20, 'Training Error=' + str(round(losses[len(losses)-1], 2)), fontsize=16)
    ax2.text(600, 50, 'Iteration=' + str(round(len(losses), 1)), fontsize=16)
 
    # グラフを表示する。
    ax1.legend()
    fig.tight_layout()
    plt.show()
    plt.close()


def main(train_loader, test_loader, *, train_datasize, test_datasize, epochs=300):
    model = ModelNet()
    # 最適化アルゴリズムと損失関数を設定
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

    with torch.no_grad():
        for x, gt_y in test_loader:
            #print("-------")
            y_pred = model(x)
            y_pred = y_pred.reshape(gt_y.shape)
            print(f"gt: {gt_y[0]} pred: {y_pred[0]}")
            #print(gt_y - y_pred)

    # print(X_train.shape, y_train.shape)
    # plot(X_train, y_train, X_test.data.numpy().T[1], y_pred, losses)


if __name__ == "__main__":
    # m = ModelNet()
    # input = torch.randn(1, 768)
    # output = m(input)
    # print(output)
    s = Path("mesh_info_15k.json").read_text()
    X, y = [], []
    for betas, perimeter in json.loads(s):
        X.append(betas)
        y.append(perimeter)

    X = torch.FloatTensor(X) * 10.0
    y = torch.FloatTensor(y) * 10.0
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_datasize = len(train_dataset)
    test_datasize = len(test_dataset)
    print(f"train_datasize={train_datasize} test_datasize={test_datasize}")
    main(train_loader, test_loader, train_datasize=train_datasize, test_datasize=test_datasize)

