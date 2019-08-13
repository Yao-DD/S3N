import os
from collections import OrderedDict
from typing import List, Dict, Tuple, Callable, Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
from PIL import Image
from nest import register


class FGVC_Dataset(Dataset):

	def __init__(self, data_dir,  split, lable_path=None, transform=None, target_transform=None):
		self.data_dir = data_dir
		self.split = split
		self.lable_path = lable_path
		self.transform = transform
		self.target_transform = target_transform

		self.image_dir = os.path.join(self.data_dir, 'images')
		self.image_lables = self._read_annotation(self.split)

	def _read_annotation(self, split):
		class_lables = OrderedDict()
		if self.lable_path is None:
			lable_path = os.path.join(self.data_dir, split + '.txt')
		else:
			lable_path = os.path.join(self.data_dir, self.lable_path, split + '.txt')
		if os.path.exists(lable_path):
			with open(lable_path, 'r') as f:
				for line in f:
					name, lable = line.split(' ')
					class_lables[name] = int(lable)
		else:
			raise NotImplementedError(
				'Invalid path for dataset')

		return list(class_lables.items())

	def __getitem__(self, index):
		filename, target = self.image_lables[index]
		img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')

		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.image_lables)


@register
def fgvc_dataset(
	split: str,
	data_dir: str,
	label_path: Optional[str] = None,
	transform: Optional[Callable] = None,
	target_transform: Optional[Callable] = None) -> Dataset:
	'''Fine-grained visual classification datasets.
	'''
	return FGVC_Dataset(data_dir, split, label_path, transform, target_transform)