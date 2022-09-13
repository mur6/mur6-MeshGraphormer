import json
import pathlib
import torch


def load1():
    template_vertices, template_3d_joints, template_vertices_sub = torch.load('template_params.pt')

    print("template_3d_joints", template_3d_joints.shape, template_3d_joints.dtype)
    print("template_vertices_sub", template_vertices_sub.shape, template_vertices_sub.dtype)
    d1 = template_3d_joints.tolist()
    d2 = template_vertices_sub.tolist()
    d = {"template_3d_joints":d1, "template_vertices_sub": d2}
    return d

def save(d, filename):
    p = pathlib.Path(filename)
    with p.open(mode='w') as fh:
        json.dump(d, fh)

from PIL import Image

import onnx
import torch.onnx
import torch
from torchvision import transforms

def load_image():
    transform = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    img = Image.open('./samples/hand/internet_fig4.jpg')
    img_tensor = transform(img)
    batch_imgs = torch.unsqueeze(img_tensor, 0)
    print(batch_imgs.shape)
    return batch_imgs.tolist()

def main():
    #d = load1()
    #save(d, "template_params.json")
    lis = load_image()
    save({"batch_imgs": lis}, "batch_imgs.json")
main()
