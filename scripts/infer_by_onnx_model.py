import argparse
from logging import DEBUG, INFO, basicConfig, debug, error, exception, getLogger, info, warning
from pathlib import Path

import onnxruntime as ort
import torch
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from torchvision import transforms

import src.modeling.data.config as cfg


from src.handinfo import utils
from src.handinfo.data.tools import make_hand_data_loader
from src.handinfo.fastmetro import get_fastmetro_model
from src.handinfo.visualize import (
    convert_mesh,
    make_hand_mesh,
    visualize_mesh,
    visualize_mesh_and_points,
)
from src.modeling._mano import MANO, Mesh

# def parse_args():
#     def parser_hook(parser):
#         parser.add_argument("--batch_size", type=int, default=4)
#         # parser.add_argument("--gamma", type=Decimal, default=Decimal("0.97"))
#         # parser.add_argument(
#         #     "--mymodel_resume_dir",
#         #     type=Path,
#         #     required=False,
#         # )

#     args = train_parse_args(parser_hook=parser_hook)
#     return args


# def test3(model_filename, image_file):
#     print(img_tensor.shape)
#     # plt.imshow(img_tensor.permute(1, 2, 0))
#     # plt.show()
#     # return
#     # # img_tensor = transform(img)
#     mano_model = MANO().to("cpu")

#     batch_imgs = torch.unsqueeze(img_tensor, 0).numpy()
#     print(batch_imgs.shape)
#     ort_sess = ort.InferenceSession(str(model_filename))
#     outputs = ort_sess.run(None, {"input": batch_imgs})
#     pred_cam, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_fine = outputs
#     # Print Result
#     print(f"pred_cam: {pred_cam}")
#     cam = pred_cam[0]
#     # K = torch.tensor([[fx, scale, tx], [0, fy, ty], [0, 0, 1]])
#     print(f"pred_3d_joints: {pred_3d_joints.shape}")
#     print(f"pred_3d_vertices_coarse: {pred_3d_vertices_coarse.shape}")
#     print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")
#     mesh = make_hand_mesh(mano_model, pred_3d_vertices_fine.squeeze(0))
#     # print(mesh)
#     visualize_mesh(mesh=mesh, cam=cam)
#     # visualize_points(points=pred_3d_joints.squeeze(0))
#     # print(pred_3d_joints.squeeze(0))



def load_image_as_tensor(image_file, show_image=False):
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_file)
    image = image.convert('RGB')

    img_tensor = transform(image)
    if show_image:
        print(img_tensor.shape)
        plt.imshow(img_tensor.permute(1, 2, 0))
        plt.show()
    return img_tensor.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--onnx_filename",
    #     type=Path,
    #     required=True,
    # )
    parser.add_argument(
        "--sample_dir",
        type=Path,
        # required=True,
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return args


def conv_camera_param(camera):
    import numpy as np

    focal_length = 1000
    # res = img.shape[1]
    res = 224
    camera_t = np.array([camera[1], camera[2], 2 * focal_length / (res * camera[0] + 1e-9)])
    return camera_t


def conv_camera_param2(camera):
    import numpy as np
    focal_length = 1000
    res = 224
    camera_t = [camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)]
    return camera_t


from trimesh.scene.cameras import Camera

def set_color(mesh, *, color):
    for facet in mesh.facets:
        # mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    return mesh
def set_blue(mesh):
    blue = [32, 32, 210, 128]
    return set_color(mesh, color=blue)
def visualize_mesh2(*, mesh, tx, ty, sc):
    import numpy as np
    cam = Camera(resolution=(640, 480), focal=(800, 800))
    # set camera parameters
    # cam.transform = trimesh.transformations.translation_matrix([tx, ty, -sc])
    # cam.projection = 'perspective'
    scene = trimesh.Scene()
    cam = np.eye(4)
    cam[3, 0:3] = [tx, ty, sc]
    scene.camera_transform = cam
    # angles=(0.5, 0, 0), distance=5,
    print(f"distance: {sc}")
    # scene.set_camera(distance=sc, center=(0,0,0))
    print(scene.camera_transform)

    scene.add_geometry(set_blue(mesh))
    scene.show()


def main(args):
    #model_filename = "onnx/gm2.onnx"
    model_filename = str(args.model_path)
    print(f"model_filename: {model_filename}")
    images = load_image_as_tensor(args.sample_dir)
    print(images.shape)
    # str(model_filename)

    ort_sess = ort.InferenceSession(model_filename)
    # outputs = ort_sess.run(None, {"images": images.numpy()})

    # (pred_camera, pred_3d_joints, pred_vertices_sub, vertices)

    (
        collision_points,
        vertices,
        faces,
        max_distance,
        min_distance,
        mean_distance,
        ring_finger_length,
        ring_finger_points,
        pred_cam,
    ) = ort_sess.run(None, {"batch_imgs": images.numpy()})

    # 'pred_camera', 'pred_3d_joints', 'pred_vertices_sub', 'pred_vertices'],
    print(f"pred_cam: {pred_cam}")
    # print(f"pred_cam:converted: {conv_camera_param(pred_cam[0])}")
    print(f"vertices: {vertices.shape}")
    # print(f"faces: {faces.shape}")

    faces = torch.load("../FastMETRO/models/weights/faces.pt")
    print(f"faces: {faces.shape}")
    ####
    camera = pred_cam
    tx = camera[1]
    ty = camera[2]
    sc = camera[0]
    debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2]}
    print(f"camera debug: {debug_text}")
    camera_t = conv_camera_param2(pred_cam)
    print(f"camera_t: {camera_t}")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    import pathlib
    export_type = "obj"
    if export_type == "obj":
        # with pathlib.Path("test.obj").open(mode="wb") as fh:
        #     # trimesh.exchange.obj.export_obj(mesh)
        #     content = trimesh.exchange.stl.export_stl(mesh)
        #     fh.write(content)
        mesh.export('jsapp/png_08.obj')
    elif export_type == "stl":
        with pathlib.Path("test.stl").open(mode="wb") as fh:
            # trimesh.exchange.obj.export_obj(mesh)
            content = trimesh.exchange.gltf.export_gltf(mesh)
            fh.write(content)
    elif export_type == "gltf":
        with pathlib.Path("test.gltf").open(mode="wb") as fh:
            # trimesh.exchange.obj.export_obj(mesh)
            content = trimesh.exchange.gltf.export_gltf(mesh)
            fh.write(content)

    # visualize_mesh2(mesh=mesh, tx=tx, ty=ty, sc=sc)


"""
PYTHONPATH=. python scripts/tests/test_infer_with_logic.py --sample_dir  demo/sample_hand_images_12/1.jpeg
"""


if __name__ == "__main__":
    args = parse_args()
    main(args)
