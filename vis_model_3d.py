from pathlib import Path
import argparse

import trimesh
import torch
import torch.nn as nn
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import torch.nn.parallel
import torch.utils.data
# from torch.autograd import Variable
import torch.nn.functional as F

from src.modeling._mano import MANO, Mesh
from test_model_3d import load_data, STN3d


def create_point_geom(ring_point):
    geom = trimesh.creation.icosphere(radius=0.0008)
    color = [202, 2, 2, 255] # red
    geom.visual.face_colors = color
    geom.apply_translation(ring_point)
    return geom

def visualize(*, mesh, points):
    color = [102, 102, 102, 64]
    for facet in mesh.facets:
        #mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    for p in points:
        scene.add_geometry(create_point_geom(p))
    scene.show()


def make_hand_mesh(gt_vertices):
    print(f"gt_vertices: {gt_vertices.shape}")
    mano_model = MANO().to("cpu")
    mano_faces = mano_model.layer.th_faces
    # mesh objects can be created from existing faces and vertex data
    return trimesh.Trimesh(vertices=gt_vertices.transpose(1, 0).numpy(), faces=mano_faces)


def infer(model, test_dataset):
    model.eval()
    with torch.no_grad():
        for x, gt_y in test_dataset:
            mesh = make_hand_mesh(x)
            print("-------")
            x = x.unsqueeze(0)
            print(f"x: {x.shape} gt_y:{gt_y.shape}")
            # print(gt_y.shape)
            y_pred = model(x)
            y_pred = y_pred.squeeze(0)
            points = y_pred.transpose(1, 0).numpy()
            print(f"points: {points.shape}")
            # y_pred = y_pred.reshape(gt_y.shape)
            # print(f"gt: {gt_y[0]} pred: {y_pred[0]}")
            visualize(mesh=mesh, points=gt_y.transpose(1, 0).numpy())
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
