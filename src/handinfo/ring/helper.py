from collections import defaultdict

import numpy as np

# from src.modeling._mano import MANO, Mesh

import src.modeling.data.config as cfg
from src.handinfo.ring.calculator import calc_perimeter_and_center_points


# def iter_meta_info(dataset_partial):
#     for img_key, transfromed_img, meta_data in dataset_partial:
#         pose = meta_data["pose"]
#         assert pose.shape == (48,)
#         betas = meta_data["betas"]
#         assert betas.shape == (10,)
#         joints_2d = meta_data["joints_2d"][:, 0:2]
#         assert joints_2d.shape == (21, 2)
#         joints_3d = meta_data["joints_3d"][:, 0:3]
#         assert joints_3d.shape == (21, 3)
#         # print(mano_pose.shape, trans.shape, betas.shape, joints_2d.shape, joints_3d.shape)
#         yield MetaInfo(pose, betas, joints_2d, joints_3d), meta_data


def _adjust_vertices(gt_vertices, gt_3d_joints):
    gt_vertices = gt_vertices / 1000.0
    gt_3d_joints = gt_3d_joints / 1000.0
    # orig_3d_joints = gt_3d_joints

    # mesh_sampler = Mesh(device=torch.device("cpu"))
    # gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
    gt_3d_root = gt_3d_joints[:, cfg.J_NAME.index("Wrist"), :]
    gt_vertices = gt_vertices - gt_3d_root[:, None, :]
    # gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
    gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
    # gt_vertices.squeeze(0), gt_3d_joints.squeeze(0)
    return gt_vertices, gt_3d_joints


MANO_JOINTS_NAME = (
    "Wrist",
    "Thumb_1",
    "Thumb_2",
    "Thumb_3",
    "Thumb_4",
    "Index_1",
    "Index_2",
    "Index_3",
    "Index_4",
    "Middle_1",
    "Middle_2",
    "Middle_3",
    "Middle_4",
    "Ring_1",
    "Ring_2",
    "Ring_3",
    "Ring_4",
    "Pinky_1",
    "Pinky_2",
    "Pinky_3",
    "Pinky_4",
)

WRIST_INDEX = MANO_JOINTS_NAME.index("Wrist")
RING_1_INDEX = MANO_JOINTS_NAME.index(f"Ring_1")
RING_2_INDEX = MANO_JOINTS_NAME.index(f"Ring_2")
RING_3_INDEX = MANO_JOINTS_NAME.index(f"Ring_3")
RING_4_INDEX = MANO_JOINTS_NAME.index(f"Ring_4")


def ring_finger_point_func(gt_3d_joints, *, num):
    """
    for
    Ring_1
    Ring_2
    """
    ring_point = MANO_JOINTS_NAME.index(f"Ring_{num}")
    return gt_3d_joints[:, ring_point, :]


def calc_ring(mano_model_wrapper, *, pose, betas):
    # print(f"pose: {pose.shape}")
    # print(f"betas: {betas.shape}")
    gt_vertices, gt_3d_joints = mano_model_wrapper.get_jv(
        pose=pose, betas=betas, adjust_func=_adjust_vertices
    )
    # print(f"gt_vertices: {gt_vertices.shape}")
    # print(f"gt_3d_joints: {gt_3d_joints.shape}")
    ring1s = ring_finger_point_func(gt_3d_joints, num=1)
    ring2s = ring_finger_point_func(gt_3d_joints, num=2)
    hand_meshes = mano_model_wrapper.get_trimesh_list(gt_vertices)
    calc_result = calc_perimeter_and_center_points(
        hand_meshes=hand_meshes, ring1s=ring1s, ring2s=ring2s
    )
    # d = dict(
    #     betas=betas.numpy(),
    #     pose=pose.numpy(),
    #     gt_3d_joints=gt_3d_joints.numpy(),
    #     gt_vertices=gt_vertices.numpy(),
    #     **calc_result
    # )
    return calc_result


def _append_im_key(img_keys, d_list):
    def _iter():
        for im_key, d in zip(img_keys, d_list):
            d["img_key"] = im_key
            yield d

    return list(_iter())


def iter_converted_batches(mano_model_wrapper, dataloader):
    for _, (img_keys, images, annotations) in enumerate(dataloader):
        pose = annotations["pose"]
        assert pose.shape[1] == 48
        betas = annotations["betas"]
        assert betas.shape[1] == 10
        # joints_2d = annotations["joints_2d"][:, 0:2]
        # # assert joints_2d.shape == (21, 2)
        # joints_3d = annotations["joints_3d"][:, 0:3]
        # # assert joints_3d.shape == (21, 3)
        # print(f"{i}, pose: {pose.shape} betas:{betas.shape}")
        d_list = calc_ring(mano_model_wrapper, pose=pose, betas=betas)
        d_list = _append_im_key(img_keys, d_list)
        # print(res[0]["perimeter"])
        yield d_list


def save_to_file(filename, output_item_list):
    def conv_to_dict(output_item_list):
        keys = (
            "perimeter",
            "radius",
            # "vert_2d",
            "vert_3d",
            # "center_points",
            # "center_points_3d",
            "pca_mean_",
            "pca_components_",
            "img_key",
        )
        output_dict = defaultdict(list)
        for item in output_item_list:
            for key in keys:
                output_dict[key].append(item[key])
        for key in keys:
            value = output_dict[key]
            output_dict[key] = np.array(value)
        return output_dict

    output_dict = conv_to_dict(output_item_list)
    np.savez(filename, **output_dict)
