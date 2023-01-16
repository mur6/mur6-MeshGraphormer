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

from test_model_3d import load_data, STN3d






def infer(model, test_dataset):
    model.eval()
    with torch.no_grad():
        for x, gt_y in test_dataset:
            print("-------")
            x = x.unsqueeze(0)
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
