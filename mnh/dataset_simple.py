# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset

# DEFAULT_DATA_ROOT = 'data'
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data"
)

def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch

class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """
    def __init__(self, entries: List):
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]


def rescale_images(images, image_size):
    scale_factors = [s_new / s for s, s_new in zip(images.shape[1:3], image_size)]
    if abs(scale_factors[0] - scale_factors[1]) > 1e-3:
        raise ValueError(
            "Non-isotropic scaling is not allowed. Consider changing the 'image_size' argument."
        )
    scale_factor = sum(scale_factors) * 0.5

    if scale_factor != 1.0:
        print(f"Rescaling dataset (factor={scale_factor})")
        images = torch.nn.functional.interpolate(
            images.permute(0, 3, 1, 2),
            size=tuple(image_size),
            mode="bilinear",
        ).permute(0, 2, 3, 1)

    return images

def get_datasets(
    data_root: str,
    dataset_name: str,  # 'lego | fern'
    image_size: Tuple[int, int]
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")

    cameras_path = os.path.join(data_root, dataset_name + ".pth")
    image_path = cameras_path.replace(".pth", ".png")

    train_data = torch.load(cameras_path)
    n_cameras = train_data["cameras"]["R"].shape[0]

    _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # The dataset image is very large ...
    images = torch.FloatTensor(np.array(Image.open(image_path))) / 255.0
    images = torch.stack(torch.chunk(images, n_cameras, dim=0))
    Image.MAX_IMAGE_PIXELS = _image_max_image_pixels

    imaegs = rescale_images(images, image_size)
    
    cameras = [
        PerspectiveCameras(
            **{k: v[cami][None] for k, v in train_data["cameras"].items()}
        ).to("cpu")
        for cami in range(n_cameras)
    ]

    train_idx, val_idx, test_idx = train_data["split"]

    train_dataset, val_dataset, test_dataset = [
        ListDataset(
            [
                {"image": images[i], "camera": cameras[i], "camera_idx": int(i)}
                for i in idx
            ]
        )
        for idx in [train_idx, val_idx, test_idx]
    ]

    return train_dataset, val_dataset, test_dataset

def get_dataset_all(
    data_root: str,
    dataset_name: str,  
    image_size: Tuple[int, int],
) -> Dataset:
    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")

    cameras_path = os.path.join(data_root, dataset_name + ".pth")
    image_path = cameras_path.replace(".pth", ".png")

    train_data = torch.load(cameras_path)
    n_cameras = train_data["cameras"]["R"].shape[0]

    _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # The dataset image is very large ...
    images = torch.FloatTensor(np.array(Image.open(image_path))) / 255.0
    images = torch.stack(torch.chunk(images, n_cameras, dim=0))
    Image.MAX_IMAGE_PIXELS = _image_max_image_pixels

    images = rescale_images(images, image_size)
    
    cameras = [
        PerspectiveCameras(
            **{k: v[cami][None] for k, v in train_data["cameras"].items()}
        ).to("cpu")
        for cami in range(n_cameras)
    ]

    dataset_all = ListDataset(
        [
            {"image": images[i], "camera": cameras[i], "camera_idx": int(i)}
            for i in range(n_cameras)
        ]
    )

    return dataset_all

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_name = 'smile_0'
    image_size = (400, 400)
    train_dataset, valid_dataset, test_dataset = get_datasets(DEFAULT_DATA_ROOT, data_name, image_size)
    print('[Dataset size] train: {} | valid: {} | test: {}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=trivial_collate
    )
    
    for i, data_batch in enumerate(train_dataloader):
        image, camera, cam_id = data_batch[0].values()
        print('image size: {}'.format(tuple(image.size())))
        print('camera: {}'.format(camera))
        print('camera index: {}'.format(cam_id))
        print(camera.image_size)
        break
    