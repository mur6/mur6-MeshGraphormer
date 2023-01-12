from pathlib import Path
import json

import torch
import torch.nn as nn
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# from models import SpatialDescriptor, StructuralDescriptor, MeshConvolution


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
        hidden_dim1 = 1024
        hidden_dim2 = 32
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim2, 1),
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

def train(model, optimizer, E, iteration, x, y):
    # 学習ループ
    losses = []
    for i in range(iteration):
        optimizer.zero_grad()                   # 勾配情報を0に初期化
        y_pred = model(x)
        loss = E(y_pred.reshape(y.shape), y)
        loss.backward()                         # 勾配の計算
        optimizer.step()                        # 勾配の更新
        losses.append(loss.item())              # 損失値の蓄積
        print('epoch=', i+1, 'loss=', loss)
    return model, losses


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


def main(X_train, X_test, y_train, y_test):
    net = ModelNet()
    # 最適化アルゴリズムと損失関数を設定
    #optimizer = optim.RMSprop(net.parameters(), lr=0.01)
    # optimizer = optim.AdamW(net.parameters(), lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    E = nn.MSELoss()
    # トレーニング
    net, losses = train(model=net, optimizer=optimizer, E=E, iteration=5000, x=X_train, y=y_train)
    y_pred = test(net, X_test)
    for gt, y in zip(y_test, y_pred):
        print(gt, y, gt-y)
    # print(X_train.shape, y_train.shape)
    # plot(X_train, y_train, X_test.data.numpy().T[1], y_pred, losses)


if __name__ == "__main__":
    # m = ModelNet()
    # input = torch.randn(1, 768)
    # output = m(input)
    # print(output)
    s = Path("mesh_info.json").read_text()
    X, y = [], []
    for betas, perimeter in json.loads(s):
        X.append(betas)
        y.append(perimeter)

    X = torch.FloatTensor(X) * 10.0
    y = torch.FloatTensor(y) * 10.0
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(X_train)
    main(X_train, X_test, y_train, y_test)

