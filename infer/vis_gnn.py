from pathlib import Path
import argparse

import trimesh
import torch
import torch.nn as nn
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader

import torch.nn.parallel
import torch.utils.data
# from torch.autograd import Variable
import torch.nn.functional as F

from src.modeling._mano import MANO, Mesh
from src.model.pointnet import PointNetfeat
from src.handinfo.data import load_data_for_geometric


def create_point_geom(ring_point, color):
    geom = trimesh.creation.icosphere(radius=0.0008)
    if color == "red":
        color = [202, 2, 2, 255]
    else:
        color = [0, 0, 200, 255]
    geom.visual.face_colors = color
    geom.apply_translation(ring_point)
    return geom


def visualize_one_point(*, mesh, a_point):
    color = [102, 102, 102, 64]
    for facet in mesh.facets:
        #mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(create_point_geom(a_point, "red"))
    scene.show()

def visualize_colored_points(*, mesh, points=[]):
    color = [102, 102, 102, 10]
    for facet in mesh.facets:
        #mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    for (p, color) in points:
        scene.add_geometry(create_point_geom(p, color))
    scene.show()

def visualize_points(*, mesh, points):
    color = [102, 102, 102, 64]
    for facet in mesh.facets:
        #mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    for p in points:
        scene.add_geometry(create_point_geom(p, "red"))
    scene.show()

def make_hand_mesh(gt_vertices):
    print(f"gt_vertices: {gt_vertices.shape}")
    mano_model = MANO().to("cpu")
    mano_faces = mano_model.layer.th_faces
    # mesh objects can be created from existing faces and vertex data
    return trimesh.Trimesh(vertices=gt_vertices, faces=mano_faces)

def similarity(x1, x2, **kwargs):
    return 1 - F.relu(F.cosine_similarity(x1, x2, **kwargs))


def infer(model, test_loader):
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            mesh = make_hand_mesh(data.pos)
            print("-------")
            output = model(data.x, data.pos, data.batch)
            print(f"output.shape: {output.shape}")
            print(f"output: {output.view(-1)}")
            mean = output.view(-1)[:3]
            normal_v = output.view(-1)[3:6]
            radius = output.view(-1)[6:]
            gt_normal_v = data.normal_v
            gt_radius = data.radius
            print(f"pred: mean: {mean}")
            print(f"gt: mean: {data.pca_mean}")
            print(f"pred: radius: {radius}")
            print(f"gt: radius: {gt_radius}")
            print(f"pred: normal_v: {normal_v}")
            print(f"gt:normal_v: {gt_normal_v}")
            print(f"類似度: {F.cosine_similarity(normal_v, gt_normal_v, 0)}")
            # cs = F.cosine_similarity(normal_v, normal_v, dim=0).abs()
            # print(cs)
            visualize_colored_points(mesh=mesh, points=[
                (mean.numpy(), "red"),
                (mean+normal_v, "blue"),
            ])
            if idx == 3:
                break


def main(resume_dir, input_filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset = load_data_for_geometric(
        input_filename,
        transform=None,
        pre_transform=None,
        device=device)
    train_datasize = len(train_dataset)
    test_datasize = len(test_dataset)
    print(f"train_datasize={train_datasize} test_datasize={test_datasize}")
    # train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    # test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    # print(train_dataset.data)
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # train_dataset, test_dataset = load_data_for_geometric(input_filename, device=device)
    # train_datasize = len(train_dataset)
    # test_datasize = len(test_dataset)
    # print(f"train_datasize={train_datasize} test_datasize={test_datasize}")
    # batch_size = 2
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if (resume_dir / "model.bin").exists() and \
        (resume_dir / "state_dict.bin").exists():
        model = torch.load(resume_dir / "model.bin", map_location=torch.device('cpu'))
        state_dict = torch.load(resume_dir / "state_dict.bin", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Model class: {model.__class__.__name__}")
        infer(model, test_loader)
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
