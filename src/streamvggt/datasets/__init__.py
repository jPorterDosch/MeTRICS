"""streamvggt multi-view datasets.

A self-contained migration of the DUSt3R dataset machinery (base classes,
sampler, cropping/correspondence/transform utilities) plus the HAMMER,
ARKitScenes and ScanNet loaders, so training no longer imports from the dust3r
tree. See :mod:`streamvggt.datasets.config` for the tyro-exposable
``DatasetConfig`` used to construct these (nested inside a training entrypoint's
config), and :mod:`streamvggt.datasets.types` for the ``Split`` / ``DatasetName``
/ ``TransformName`` enums.
"""

from .arkitscenes import ARKitScenes_Multi
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .base.base_multiview_dataset import EmptyDatasetError
from .base.batched_sampler import BatchedRandomSampler
from .base.easy_dataset import CatDataset
from .config import DatasetConfig, MultiDatasetConfig, build_dataset
from .types import DatasetName, Split, TransformName
from .hammer import HAMMER_Multi
from .hypersim import HyperSim_Multi
from .scannet import ScanNet_Multi

import torch


__all__ = [
    "ARKitScenes_Multi",
    "ARKitScenesHighRes_Multi",
    "HAMMER_Multi",
    "HyperSim_Multi",
    "ScanNet_Multi",
    "BatchedRandomSampler",
    "CatDataset",
    "DatasetConfig",
    "EmptyDatasetError",
    "MultiDatasetConfig",
    "build_dataset",
    "DatasetName",
    "Split",
    "TransformName",
    "get_data_loader",
]


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator=None,
    fixed_length=False,
    seed=None,
):
    """Wrap an already-constructed multi-view dataset in a DataLoader driven by
    its aspect-ratio-aware batched sampler.

    The dataset must be a real object (build it with ``DatasetConfig.build()``),
    not a string -- there is no ``eval`` here. A dataset without ``make_sampler``
    raises rather than silently falling back to a plain DataLoader, so a wiring
    mistake surfaces immediately instead of training on the wrong sampler.

    ``seed`` pins worker seeding (and thus per-sample aug/stride draws) to a
    dedicated generator instead of the global torch RNG, so the train stream is
    identical across runs regardless of what the model consumed at init.
    """

    sampler = dataset.make_sampler(
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        world_size=1 if accelerator is None else accelerator.num_processes,
        fixed_length=fixed_length,
    )
    # TODO: this pairs only the data stream across arms; model-side draws
    # (dropout, MAE masking) stay unpaired. Revisit if we want paired masks.
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        generator=generator,
    )
