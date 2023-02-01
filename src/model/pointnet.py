import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Simple_STN3d(nn.Module):
    in_features = 778

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 778, 1)
        self.conv2 = torch.nn.Conv1d(778, 1024, 1)
        self.conv3 = torch.nn.Conv1d(1024, 2048, 1)
        self.max_out = 2048
        fc1_out, fc2_out = 512, 256
        self.fc1 = nn.Linear(self.max_out, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 6)
        # self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(778)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(self.max_out)
        # ---
        self.bn4 = nn.BatchNorm1d(fc1_out)
        self.bn5 = nn.BatchNorm1d(fc2_out)

    def forward(self, x):
        batch_size = x.size()[0]
        # print(f"x: {x.size()} batch_size: {batch_size}")
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print("STN3d 0:", x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        # print("STN3d 1:", x.shape)
        x = x.view(-1, self.max_out)
        # print("STN3d 2:", x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        # print("STN3d 3:", x.shape)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x


class STN3d(nn.Module):
    # in_features = 778

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 778, 1)
        self.conv2 = torch.nn.Conv1d(778, 1024, 1)
        self.conv3 = torch.nn.Conv1d(1024, 2048, 1)
        self.max_out = 2048
        fc1_out, fc2_out = 512, 256
        self.fc1 = nn.Linear(self.max_out, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 9)
        # self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(778)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(self.max_out)
        # ---
        self.bn4 = nn.BatchNorm1d(fc1_out)
        self.bn5 = nn.BatchNorm1d(fc2_out)

    def forward(self, x):
        batch_size = x.size()[0]
        # print(f"x: {x.size()} batch_size: {batch_size}")
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print("STN3d 0:", x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        # print("STN3d 1:", x.shape)
        x = x.view(-1, self.max_out)
        # print("STN3d 2:", x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        # print("STN3d 3:", x.shape)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # print("STN3d 4:", x.shape)
        iden = Variable(torch.eye(3, dtype=torch.float32).view(1, 9)).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        # print("STN3d 5:", x.shape)
        return x


class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.output_feature_num = 7
        self.conv3 = torch.nn.Conv1d(128, self.output_feature_num, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.output_feature_num)

    def forward(self, x):
        n_pts = x.size()[2]
        # print(f"n_pts: {n_pts}")
        trans = self.stn(x)
        # print(f"trans: {trans.shape}")
        x = x.transpose(2, 1)
        # print(f"x: {x.shape}")
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
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
