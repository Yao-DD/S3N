import os
from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from nest import register


@register
def image_transform(
    image_size: Union[int, List[int]],
    augmentation: dict,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]) -> Callable:
    """Image transforms.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    horizontal_flip = augmentation.pop('horizontal_flip', None)
    if horizontal_flip is not None:
        assert isinstance(horizontal_flip, float) and 0 <= horizontal_flip <= 1

    vertical_flip = augmentation.pop('vertical_flip', None)
    if vertical_flip is not None:
        assert isinstance(vertical_flip, float) and 0 <= vertical_flip <= 1

    random_crop = augmentation.pop('random_crop', None)
    if random_crop is not None:
        assert isinstance(random_crop, dict)

    center_crop = augmentation.pop('center_crop', None)
    if center_crop is not None:
        assert isinstance(center_crop, (int, list))

    if len(augmentation) > 0:
        raise NotImplementedError('Invalid augmentation options: %s.' % ', '.join(augmentation.keys()))
    
    t = [
        transforms.Resize(image_size) if random_crop is None else transforms.RandomResizedCrop(image_size[0], **random_crop),
        transforms.CenterCrop(center_crop) if center_crop is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    
    return transforms.Compose([v for v in t if v is not None])


@register
def fetch_data(
    dataset: Callable[[str], Dataset],
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    train_splits: List[str] = [],
    test_splits: List[str] = [],
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    test_image_size: int = 600, 
    train_augmentation: dict = {},
    test_augmentation: dict = {},
    batch_size: int = 1,
    test_batch_size: Optional[int] = None) -> Tuple[List[Tuple[str, DataLoader]], List[Tuple[str, DataLoader]]]:
    """Fetch data.
    """

    train_transform = transform(augmentation=train_augmentation) if transform else None
    train_loader_list = []
    for split in train_splits:
        train_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split, 
                transform = train_transform,
                target_transform = target_transform),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = train_shuffle)))
    
    test_transform = transform(image_size=[test_image_size, test_image_size], augmentation=test_augmentation) if transform else None
    test_loader_list = []
    for split in test_splits:
        test_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split, 
                transform = test_transform,
                target_transform = target_transform),
            batch_size = batch_size if test_batch_size is None else test_batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = test_shuffle)))

    return train_loader_list, test_loader_list