from logging import getLogger

import torch

from src.datasets.build import make_hand_data_loader as make_freihand_data_loader
from src.datasets.build import make_batch_data_sampler, make_data_sampler
from src.datasets.my_dataset import BlenderHandMeshDataset
from src.utils.comm import all_gather, get_rank, get_world_size


logger = getLogger(__name__)


def make_blender_data_loader(
    args, *, blender_ds_base_path, is_distributed=True, is_train=True, start_iter=0, scale_factor=1
):
    dataset = BlenderHandMeshDataset(base_path=blender_ds_base_path)

    shuffle = True
    images_per_gpu = args.per_gpu_train_batch_size
    images_per_batch = images_per_gpu * get_world_size()
    # print(f"images_per_batch: {images_per_batch}")
    # print(f"dataset count: {len(dataset)}")
    iters_per_batch = len(dataset) // images_per_batch
    print(f"iters_per_batch: {iters_per_batch}")
    num_iters = iters_per_batch * args.num_train_epochs
    logger.info("Train with {} images per GPU.".format(images_per_gpu))
    logger.info("Total batch size {}".format(images_per_batch))
    logger.info("Total training steps {}".format(num_iters))

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(sampler, images_per_gpu, num_iters, start_iter)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader


def make_train_hand_data_loader(args, *, is_distributed, scale_factor):
    train_yaml = args.train_yaml
    blender_ds_base_path = args.blender_ds_base_path
    print(f"Debug: train_yaml={train_yaml}, blender_ds_base_path={blender_ds_base_path}")
    if train_yaml and blender_ds_base_path:
        raise RuntimeError(f"Both train_yaml and blender_ds_base_path are specified. Please specify just one.")

    if train_yaml:
        train_dataloader = make_freihand_data_loader(
            args, args.train_yaml, is_distributed, is_train=True, scale_factor=scale_factor
        )
        dataset_type = "freihand"
    elif blender_ds_base_path:
        train_dataloader = make_blender_data_loader(
            args,
            blender_ds_base_path=blender_ds_base_path,
            is_distributed=False,
            is_train=True,
            scale_factor=scale_factor)
        dataset_type = "blender"
    else:
        raise RuntimeError(f"Please specify either `train_yaml` or `blender_ds_base_path`.")
    return train_dataloader, dataset_type
