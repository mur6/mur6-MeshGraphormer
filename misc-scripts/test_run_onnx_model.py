import logging
from logging import getLogger, basicConfig

from src.modeling.bert.e2e_hand_network import generate_t_pose_template_mesh, get_template_vertices_sub, template_normalize

from PIL import Image

import onnx
import torch.onnx
import torch
from torchvision import transforms

basicConfig(level=logging.INFO)
logger = getLogger(__name__)

transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
img = Image.open('./samples/hand/internet_fig4.jpg')
img_tensor = transform(img)
print(img_tensor.shape)
img_tensor = transform(img)
batch_imgs = torch.unsqueeze(img_tensor, 0)
#template_vertices, template_3d_joints = generate_t_pose_template_mesh(mano)
#template_vertices_sub = get_template_vertices_sub(mesh_sampler, template_vertices)
#template_vertices, template_3d_joints, template_vertices_sub = template_normalize(template_vertices, template_3d_joints, template_vertices_sub)
template_vertices, template_3d_joints, template_vertices_sub = torch.load('template_params.pt')
#print(template_vertices.shape)

import onnxruntime as ort
import numpy as np

"""
graph torch-jit-export (
  %batch_imgs[FLOAT, 1x3x224x224]
  %template_3d_joints[FLOAT, 1x21x3]
  %template_vertices_sub[FLOAT, 1x195x3]
) initializers (
"""
ort_sess = ort.InferenceSession('gm2.onnx')
outputs = ort_sess.run(None, {"batch_imgs": batch_imgs.numpy(), "template_3d_joints": template_3d_joints.numpy(), "template_vertices_sub": template_vertices_sub.numpy()})
pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att, *rest = outputs
# Print Result 
print(pred_camera)
print(pred_vertices.shape)
print(pred_vertices)
