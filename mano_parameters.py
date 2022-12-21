import numpy as np
import torch
#from src.modeling._mano import MANO
from manopth.manolayer import ManoLayer

# Mesh and SMPL utils
# mano_model = MANO().to("cpu")
# mano_model.layer = mano_model.layer.cuda()


def calc_vert_and_joint(pose, betas):
    assert pose.shape == (1, 48)
    assert betas.shape == (1, 10)
    mano_layer = ManoLayer(
        mano_root='src/modeling/data',
        flat_hand_mean=True,
        side='right',
        use_pca = False,
        #ncomps = 45,
        # root_rot_mode='rotmat',
        # joint_rot_mode='rotmat',
        #robust_rot=True
    ) # load right hand MANO model
    gt_vertices, gt_3d_joints = mano_layer(pose, betas)
    gt_vertices.squeeze_(0)
    gt_3d_joints.squeeze_(0)
    print(f"vertices.shape: {gt_vertices.shape}, 3d_joints.shape:{gt_3d_joints.shape}")
    print(f"vertices[0]: {gt_vertices[0:4]}")
    print(f"3d_joints[0]: {gt_3d_joints[0:4]}")
    return gt_vertices, gt_3d_joints


def make_pose_and_betas(*, fill_value=0.0):
    pose = torch.full((1, 45), fill_value=fill_value)
    pose = torch.cat((pose, torch.zeros((1, 3))), dim=1)
    print(pose)
    # print("### k1:", pose[:, :6])
    #pose[:, :6] = torch.zeros((1, 6))
    # print("### k2:", pose)
    betas = torch.full((1, 10), fill_value=fill_value)
    return pose, betas


def generate_and_save(fill_value, filename):
    pose, betas = make_pose_and_betas(fill_value=fill_value)
    vert, joint = calc_vert_and_joint(pose, betas)
    np.savez(filename, vert=vert.numpy(), joint=joint.numpy())


def main():
    for i in range(7 + 1):
        fill_value = i * 0.1
        print(f"fill_value={fill_value}")
        generate_and_save(fill_value=fill_value, filename=f"mesh_gra_zero_point_{i}")


# def test_2():
#     batch_size = 1
#     random_shape = torch.rand(batch_size, 10)
#     print(random_shape.shape)
#     random_pose = torch.rand(batch_size, 45 + 3)
#     print(random_pose.shape)
#     gt_vertices, gt_3d_joints = mano_model.layer(random_pose, random_shape)

if __name__ == "__main__":
    main()
