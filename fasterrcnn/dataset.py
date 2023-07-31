import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision


class Compose:
	def __init__(self, transforms):
		self.transforms = transforms
	def __call__(self, image, target):
		for t in self.transforms:
			image = t(image)
		return image, target

class CustomDataset(Dataset):
	def __init__(self, image_dir, annotation_dir, class_dict, transforms=None):
		self.image_dir = image_dir
		self.annotation_dir = annotation_dir
		self.class_dict = class_dict
		self.transforms = transforms		
		self.image_files = os.listdir(image_dir)

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		image_name = self.image_files[idx]
		image_path = os.path.join(self.image_dir, image_name)
		annotation_path = os.path.join(self.annotation_dir, image_name.replace(".jpg", ".txt"))

		image = Image.open(image_path).convert("RGB")

		# Load YOLO annotations from the text file
		with open(annotation_path, "r") as f:
			lines = f.readlines()

		boxes = []
		labels = []
		for line in lines:
			class_index, x0, y0, x1, y1 = [float(x) for x in line.strip().split(",")]
			class_index = int(class_index)
			boxes.append([x0, y0, x1, y1])
			labels.append(class_index)

		if (len(boxes) == 0):
			boxes = torch.empty((0, 4), dtype=torch.float32)
		else:
			boxes = torch.tensor(boxes, dtype=torch.float32)
			boxes = torch.squeeze(boxes, 1)


		labels = torch.ones((len(boxes),), dtype=torch.int64)
	


		target = {}
		target["boxes"] = boxes
		target["labels"] = labels


		#if self.transforms is not None:
		#	image, target = self.transforms(image, target)


		image=torchvision.transforms.ToTensor()(image)


		return image, target