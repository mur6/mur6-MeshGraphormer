from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_cluster import fps, knn_graph

from timm.scheduler import CosineLRScheduler
from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
from torch_geometric.utils import scatter

from src.handinfo.data import load_data_for_geometric
from src.model.transformer import ClassificationNet



def save_checkpoint(model, epoch, iteration=None):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoint-{epoch}"
    checkpoint_dir.mkdir(exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model

    torch.save(model_to_save, checkpoint_dir / "model.bin")
    torch.save(model_to_save.state_dict(), checkpoint_dir / "state_dict.bin")
    print(f"Save checkpoint to {checkpoint_dir}")
    return checkpoint_dir



transform = T.Compose([
    T.RandomJitter(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2),
])
pre_transform = T.NormalizeScale()

# train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
#                          pre_transform=pre_transform)
# test_dataset = ShapeNet(path, category, split='test',
#                         pre_transform=pre_transform)
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


def main(filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = load_data_for_geometric(
        filename,
        transform=transform,
        pre_transform=pre_transform,
        device=device)
    train_datasize = len(train_dataset)
    test_datasize = len(test_dataset)
    print(f"train_datasize={train_datasize} test_datasize={test_datasize}")

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = ClassificationNet(
        in_channels=3,
        out_channels=3,
        dim_model=[32, 64, 128, 256, 512],
        ).to(device)
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    ####### test:
    # for d in train_loader:
    #     print(d.x.shape)
    #     output = model(d.x, d.pos, d.batch)
    #     print(output.shape)
    #     break


    for epoch in range(1, 1000 + 1):
        train(model, epoch, train_loader, train_datasize, optimizer, scheduler, device)
        test(model, test_loader, test_datasize, device)
        if epoch % 5 == 0:
            save_checkpoint(model, epoch)
        scheduler.step()
        # train(model, device, train_loader, optimizer)
        # iou = test(model, device, test_loader)
        # print(f'Epoch: {epoch:03d}, Test IoU: {iou:.4f}')
        # scheduler.step()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    main(filename)
