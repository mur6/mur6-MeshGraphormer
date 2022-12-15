import torch

from manopth import rodrigues_layer
from scipy.spatial.transform import Rotation as R


def th_posemap_axisang(pose_vectors):
    print(f"##### th_posemap_axisang: [input] pose_vectors.shape={pose_vectors.shape}")
    print(pose_vectors)
    rot_nb = int(pose_vectors.shape[1] / 3)
    pose_vec_reshaped = pose_vectors.contiguous().view(-1, 3)
    rot_mats = rodrigues_layer.batch_rodrigues(pose_vec_reshaped)
    rot_mats = rot_mats.view(pose_vectors.shape[0], rot_nb * 9)
    # pose_maps = subtract_flat_id(rot_mats)
    # print(f"##### th_posemap_axisang: pose_maps.shape={pose_maps.shape}, rot_mats.shape={rot_mats.shape}")
    # print(f"pose_maps: {pose_maps}")
    # print(f"rot_mats: {rot_mats}")
    # return pose_maps, rot_mats

def main():
    a = torch.FloatTensor([[0.5774, 0.5774, 0.5774]])
    rot_mats = rodrigues_layer.batch_rodrigues(a)
    print(rot_mats)

def test_scipy_rodrigues_1():
    r = R.from_mrp([0.5774, 0.5774, 0.5774])
    # a = r.as_euler('xyz', degrees=True)
    # print(a)
    # print(r.as_quat())
    print(r.as_matrix())
    r = R.from_matrix([[
        [-1/3, 2/3, 2/3],
        [2/3, -1/3, 2/3],
        [2/3, 2/3, -1/3]]
    ])
    print(r.as_mrp())
    print(r.as_euler('xyz', degrees=True))


def test_scipy_rodrigues_2():
    r = R.from_mrp([1.0667, -0.6252,  1.8175])
    print("euler:", r.as_euler('xyz', degrees=True))
    print("rodrigues:", r.as_mrp())
    # print(r.as_quat())
    print(r.as_matrix())


test_scipy_rodrigues_2()
print("############")
#main()
