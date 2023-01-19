from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

loss_chamfer, _ = chamfer_distance(ori_data, adv_pl)
loss_edge = mesh_edge_loss(new_defrom_mesh)
loss_laplacian = mesh_laplacian_smoothing(new_defrom_mesh, method="uniform")

# loss = adv_loss + torch.from_numpy(current_weight).mean()*(loss_chamfer * w_chamfer + loss_edge * w_edge  + loss_laplacian * w_laplacian)
