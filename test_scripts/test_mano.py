from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch import nn, optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from timm.scheduler import CosineLRScheduler

from src.modeling._mano import MANO
import trimesh

if __name__ == "__main__":
    mano_model = MANO()
    device = "cpu"
    if device == "cuda":
        model.to(device)
        mano_model = MANO().to(device)
        mano_model.layer = mano_model.layer.cuda()
    mano_faces = mano_model.layer.th_faces
    print(mano_faces)
