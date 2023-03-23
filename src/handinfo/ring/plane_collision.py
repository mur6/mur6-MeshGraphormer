# import trimesh
import torch
from torch import nn

from src.handinfo.ring.helper import RING_1_INDEX, RING_4_INDEX, WRIST_INDEX


class PlaneCollision:
    @staticmethod
    def cut_sub_faces(faces, *, begin_index, offset):
        A = faces - begin_index
        B = ((0 <= A) & (A < offset)).all(axis=1)
        return A[B]

    @staticmethod
    def ring_finger_submesh(mesh_vertices, mesh_faces):
        begin_index = 468
        offset = 112
        new_vertices = mesh_vertices[begin_index : begin_index + offset]
        new_faces = PlaneCollision.cut_sub_faces(mesh_faces, begin_index=begin_index, offset=offset)
        return new_vertices, new_faces

    # @staticmethod
    # def ring_finger_triangles_as_tensor(ring_mesh):
    #     def _iter_ring_mesh_triangles():
    #         for face in ring_mesh.faces:
    #             vertices = ring_mesh.vertices
    #             vertices_of_triangle = torch.from_numpy(vertices[face]).float()
    #             yield vertices_of_triangle

    #     triangles = list(_iter_ring_mesh_triangles())
    #     return torch.stack(triangles)

    def __init__(self, ring_finger_triangles, *, pca_mean, normal_v):
        # self.ring_mesh = PlaneCollision.ring_finger_submesh(mesh_vertices, mesh_faces)
        self.ring_finger_triangles = ring_finger_triangles
        # self.ring_finger_triangles = PlaneCollision.ring_finger_triangles_as_tensor(self.ring_mesh)
        self.pca_mean = pca_mean
        self.normal_v = normal_v

    def get_triangle_sides(self):
        inputs = self.ring_finger_triangles
        shifted = torch.roll(input=inputs, shifts=1, dims=1)
        stacked_sides = torch.stack((inputs, shifted), dim=2)
        return stacked_sides

    def get_inner_product_signs(self, triangle_sides):
        inner_product = triangle_sides @ self.normal_v
        inner_product_sign = inner_product[:, :, 0] * inner_product[:, :, 1]
        return inner_product_sign

    def get_collision_points(self, triangle_sides):
        plane_normal = self.normal_v
        plane_point = torch.zeros(3, dtype=torch.float32)
        line_endpoints = triangle_sides
        ray_point = line_endpoints[:, 0, :]
        ray_direction = line_endpoints[:, 1, :] - line_endpoints[:, 0, :]
        n_dot_u = ray_direction @ plane_normal
        w = ray_point - plane_point
        si = -(w @ plane_normal) / n_dot_u
        si = si.unsqueeze(1)
        collision_points = w + si * ray_direction + plane_point
        return collision_points

    def get_filtered_collision_points(self, *, sort_by_angle):
        triangle_sides = self.get_triangle_sides() - self.pca_mean
        signs = self.get_inner_product_signs(triangle_sides).view(-1)
        # print(f"triangle_sides: {triangle_sides.shape}")
        triangle_sides = triangle_sides.view(-1, 2, 3)
        collision_points = self.get_collision_points(triangle_sides)
        points = collision_points[signs <= 0]
        if sort_by_angle:
            ref_vec = points[0]
            cosines = torch.matmul(points, ref_vec) / (
                torch.norm(ref_vec) * torch.norm(points, dim=1)
            )
            angles = torch.acos(cosines)
            # print("cosines:", cosines.shape)
            # print("angles:", angles.shape)
            # for cos, ang in zip(cosines, angles):
            #     print(cos, ang)
            return points[torch.argsort(angles)][::2]  # 重複も除く
        else:
            return points


# epsilon = 1e-6


def _get_3d_joints(vertices):
    """
    This method is used to get the joint locations from the SMPL mesh
    Input:
        vertices: size = (B, 778, 3)
    Output:
        3D joints: size = (B, 21, 3)
    """
    joint_regressor_torch = torch.load("models/weights/joint_regressor.pt")
    joints = torch.einsum("bik,ji->bjk", [vertices, joint_regressor_torch])
    return joints


def make_plane_normal_and_origin_from_3d_vertices(pred_3d_joints, pred_3d_vertices_fine):
    pred_3d_joints_from_mano = _get_3d_joints(pred_3d_vertices_fine)
    pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:, WRIST_INDEX, :]
    pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
    pred_3d_joints = pred_3d_joints - pred_3d_joints_from_mano_wrist[:, None, :]
    ring1_point = pred_3d_joints[:, 13, :]
    ring2_point = pred_3d_joints[:, 14, :]
    plane_normal = ring2_point - ring1_point  # (batch X 3)
    plane_origin = (ring1_point + ring2_point) / 2  # (batch X 3)
    return pred_3d_joints, pred_3d_vertices_fine, plane_normal, plane_origin


class WrapperForRadiusModel(nn.Module):
    def __init__(self, fastmetro_model, mesh_sampler, faces):
        super().__init__()
        self.fastmetro_model = fastmetro_model
        self.mesh_sampler = mesh_sampler
        self.faces = faces
        # print(f"faces: {self.faces.dtype}")

    def forward(self, images):
        (
            pred_cam,
            pred_3d_joints,
            pred_3d_vertices_coarse,
            pred_3d_vertices_fine,
            cam_features,
            enc_img_features,
            jv_features,
        ) = self.fastmetro_model(images)

        # vertices = pred_3d_vertices_fine.squeeze(0)

        # mesh_vertices = torch.load("vertices.pt")
        # pred_3d_joints = torch.load("pred_3d_joints.pt")
        # pred_3d_vertices_fine = torch.load("pred_3d_vertices_fine.pt")
        # print(f"mesh_faces: {mesh_faces.shape}")
        # print(f"pred_3d_joints: {pred_3d_joints.shape}")
        # print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")

        (
            pred_3d_joints,
            pred_3d_vertices_fine,
            plane_normal,
            plane_origin,
        ) = make_plane_normal_and_origin_from_3d_vertices(pred_3d_joints, pred_3d_vertices_fine)

        ring_mesh_vertices, ring_mesh_faces = PlaneCollision.ring_finger_submesh(
            pred_3d_vertices_fine[0], self.faces
        )
        ring_finger_triangles = ring_mesh_vertices[ring_mesh_faces].float()

        plane_colli = PlaneCollision(
            ring_finger_triangles, pca_mean=plane_origin[0], normal_v=plane_normal[0]
        )
        collision_points = plane_colli.get_filtered_collision_points(sort_by_angle=True)
        # 1: 薬指根本周囲の点群の、原点からの距離を計算
        distance_from_origin = torch.norm(collision_points, dim=1)
        max_distance = distance_from_origin.max()
        min_distance = distance_from_origin.min()
        mean_distance = distance_from_origin.mean()
        # 2: 薬指根本周囲の点を、原点位置から元の位置に移動
        collision_points = collision_points + plane_origin
        # 3: 薬指の長さを計算
        joints = pred_3d_joints[0]
        # joints[RING_1_INDEX : RING_4_INDEX + 1]
        ring_finger_length = torch.norm(joints[RING_1_INDEX] - joints[RING_4_INDEX])
        ring_finger_points = joints[[RING_1_INDEX, RING_4_INDEX]]
        # 4:
        vertices = pred_3d_vertices_fine[0]  # [self.faces]
        if True:
            return (
                collision_points,
                vertices,  # pred_3d_vertices_fine[0],
                self.faces,
                max_distance,
                min_distance,
                mean_distance,
                ring_finger_length,
                ring_finger_points,
                pred_cam,
            )
        else:
            return (
                plane_origin,
                plane_normal,
                collision_points,
                pred_3d_joints,
                pred_3d_vertices_fine,
            )
