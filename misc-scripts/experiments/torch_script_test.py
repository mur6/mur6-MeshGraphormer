import torch
# TorchScriptを直接記述してしまう
@torch.jit.script
def test_sample(tensor):
    return torch.sum(tensor)



@torch.jit.script
def calc_features(images, j_name_index_wrist, template_vertices, template_3d_joints, template_vertices_sub):
    batch_size = images.size(0)
    # Generate T-pose template mesh

    # normalize
    ## cfg.J_NAME.index('Wrist')
    template_root = template_3d_joints[:,j_name_index_wrist,:]
    template_3d_joints = template_3d_joints - template_root[:, None, :]
    template_vertices = template_vertices - template_root[:, None, :]
    template_vertices_sub = template_vertices_sub - template_root[:, None, :]
    num_joints = template_3d_joints.shape[1]

    # concatinate template joints and template vertices, and then duplicate to batch size
    ref_vertices = torch.cat([template_3d_joints, template_vertices_sub],dim=1)
    ref_vertices = ref_vertices.expand(batch_size, -1, -1)
    return images
    # extract grid features and global image features using a CNN backbone
    image_feat, grid_feat = self.backbone(images)
    # concatinate image feat and mesh template
    image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
    # process grid features
    grid_feat = torch.flatten(grid_feat, start_dim=2)
    grid_feat = grid_feat.transpose(1,2)
    grid_feat = self.grid_feat_dim(grid_feat)
    # concatinate image feat and template mesh to form the joint/vertex queries
    features = torch.cat([ref_vertices, image_feat], dim=2)
    # prepare input tokens including joint/vertex queries and grid features
    features = torch.cat([features, grid_feat],dim=1)
    return features
# TorchScriptの内容をPythonライクに表示
print(calc_features.code)
# TorchScriptの内容を内部graph表現で表示
print(calc_features.graph)

@torch.jit.script
def test(x: torch.Tensor, n1: int):
    #if x.ndimension() < 3:
    return x + n1
print(test.code)
print(test.graph)

@torch.jit.script
def test2(x: torch.Tensor, n1: int):
    if x.ndim < 3:
        return x + n1
    else:
        return x


print(test2.code)
print(test2.graph)

@torch.jit.script
def test3(n1: int):
    a = list(range(0, n1))
    a.reverse()
    for i in a:
        
    return torch.tensor(a)

print(test3.code)
print(test3.graph)

