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


def main_back(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh_sampler = Mesh(device=device)
    mano_model = MANO().to("cpu")

    fastmetro_model = get_fastmetro_model(
        args, mesh_sampler=mesh_sampler, force_from_checkpoint=True
    )
    inputs = torch.randn(16, 3, 224, 224)
    # out = fastmetro_model(inputs)
    # print(len(out))
    model = WrapperForRadiusModel(fastmetro_model)
    (
        plane_origin,
        plane_normal,
        _,
        pred_3d_joints,
        pred_3d_vertices_fine,
    ) = model(inputs, mano_model)
    # model.eval()
    print(plane_origin, plane_normal)


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
    args = parser.parse_args()
    return args


def conv_camera_param(camera):
    import numpy as np

    focal_length = 1000
    # res = img.shape[1]
    res = 224
    camera_t = np.array([camera[1], camera[2], 2 * focal_length / (res * camera[0] + 1e-9)])
    return camera_t


def main(args):
    images = load_image_as_tensor(args.sample_dir)
    print(images.shape)
    # str(model_filename)
    model_filename = "onnx/radius_pred_cam_model.onnx"
    ort_sess = ort.InferenceSession(model_filename)
    # outputs = ort_sess.run(None, {"images": images.numpy()})
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
    ) = ort_sess.run(None, {"images": images.numpy()})
    print(f"pred_cam: {pred_cam}")
    print(f"pred_cam:converted: {conv_camera_param(pred_cam[0])}")
    # print(f"vertices: {vertices.shape}")
    # print(f"faces: {faces.shape}")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    visualize_mesh(mesh=mesh)


if __name__ == "__main__":
    args = parse_args()
    main(args)
