import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    in_features = 778

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 778, 1)
        self.conv2 = torch.nn.Conv1d(778, 1024, 1)
        self.conv3 = torch.nn.Conv1d(1024, 2048, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        # self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(778)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        # print(f"x: {x.size()} batch_size: {batch_size}")
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
        iden = Variable(torch.eye(3, dtype=torch.float32).view(1, 9)).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        # self.feature_transform = feature_transform
        # if self.feature_transform:
        #     self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        print(x.shape)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # if self.feature_transform:
        #     trans_feat = self.fstn(x)
        #     x = x.transpose(2,1)
        #     x = torch.bmm(x, trans_feat)
        #     x = x.transpose(2,1)
        # else:
        #     trans_feat = None

        # pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # if self.global_feat:
        #     return x, trans, trans_feat
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        #     return torch.cat([x, pointfeat], 1), trans,
        return x

if __name__ == "__main__":
    # input = torch.randn(1, 3, 778)
    # m = torch.nn.Conv1d(3, 778, 1)
    m = PointNetfeat()
    m.eval()
    input = torch.randn(1, 3, 778)
    output = m(input)
    print(output.shape)
    # main(args.resume_dir, args.input_filename)
