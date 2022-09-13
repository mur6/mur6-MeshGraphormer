#self.mano_dir
#self.layer = self.get_layer()
import pathlib

import numpy as np
import torch
import torch.nn as nn


mano_root = pathlib.Path('src/modeling/data')

from manopth.manolayer import ManoLayer

layer = ManoLayer(mano_root=mano_root, flat_hand_mean=False, use_pca=False)
#joint_regressor = layer.th_J_regressor.numpy()

def make_joint_regressor_torch(layer):
    joint_regressor = layer.th_J_regressor.numpy()
    # joint_num = 21
    # joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
    # skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
    # self.root_joint_idx = self.joints_name.index('Wrist')

    # # add fingertips to joint_regressor
    # fingertip_vertex_idx = [745, 317, 444, 556, 673] # mesh vertex idx (right hand)
    thumbtip_onehot = np.array([1 if i == 745 else 0 for i in range(joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
    indextip_onehot = np.array([1 if i == 317 else 0 for i in range(joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
    middletip_onehot = np.array([1 if i == 445 else 0 for i in range(joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
    ringtip_onehot = np.array([1 if i == 556 else 0 for i in range(joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
    pinkytip_onehot = np.array([1 if i == 673 else 0 for i in range(joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
    joint_regressor = np.concatenate((joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
    joint_regressor = joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]
    joint_regressor_torch = torch.from_numpy(joint_regressor).float()
    return joint_regressor_torch

def get_3d_joints(joint_regressor_torch, vertices):
    joints = torch.einsum('bik,ji->bjk', [vertices, joint_regressor_torch])
    return joints

joint_regressor_torch = make_joint_regressor_torch(layer)
print(f"joint_regressor_torch: {joint_regressor_torch.shape}")
#vertices = torch.zeros(10, 778, 3)
f = "/Users/taichi.muraki/workspace/programming/MANO/vertices_temp.pt"
vertices = torch.load(f)
joints = get_3d_joints(joint_regressor_torch, vertices)
print(f"joints: {joints.shape}")
torch.save(joints, "/Users/taichi.muraki/workspace/programming/MANO/joints.pt")
print("done.")
