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

from test_model_3d import load_data



class STN3d(nn.Module):
    in_features = 768

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
        x = self.fc3(x)
        # iden = Variable(torch.eye(3, dtype=torch.float32).view(1, 9)).repeat(batch_size, 1)
        # x = x + iden
        x = x.view(-1, 3, 20)
        return x


def infer(model, test_dataset):
    with torch.no_grad():
        for x, gt_y in test_dataset:
            print("-------")
            print(x.shape)
            print(gt_y.shape)
            y_pred = model(x)
            print(y_pred)
            # y_pred = y_pred.reshape(gt_y.shape)
            # print(f"gt: {gt_y[0]} pred: {y_pred[0]}")
            break



def main(resume_dir, input_filename):
    train_dataset, test_dataset = load_data(input_filename)
    if (resume_dir / "model.bin").exists() and \
        (resume_dir / "state_dict.bin").exists():
        model = torch.load(resume_dir / "model.bin")
        state_dict = torch.load(resume_dir / "state_dict.bin")
        model.load_state_dict(state_dict)
        infer(model, test_dataset)
    else:
        raise Exception(f"{resume_dir} is not valid directory.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_dir",
        type=Path,
        required=True,
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
