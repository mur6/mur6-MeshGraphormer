import torch
import torch.nn as nn
# from models import SpatialDescriptor, StructuralDescriptor, MeshConvolution


class SpatialDescriptor(nn.Module):

    def __init__(self):
        super(SpatialDescriptor, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, centers):
        return self.spatial_mlp(centers)

# m = SpatialDescriptor()
# input = torch.randn(1, 3, 10)
# # x = torch.FloatTensor((1, 3, 10))
# output = m(input)
# print(output.shape)

class ModelNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.in_features =   self.backbone.num_features
        in_features = 768
        hidden_dim = 256
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, input):
        y = self.head(input)
        return y

if __name__ == "__main__":
    m = ModelNet()
    input = torch.randn(1, 768)
    output = m(input)
    print(output.shape)
