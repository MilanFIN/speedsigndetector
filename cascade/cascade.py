import numpy as np
import cv2


import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F


speedSigns = [  "20",
				"30",
				"40",
				"50",
				"60",
				"70",
				"80",
				"100",
				"120"]

class SignDataset(Dataset):
	def __init__(self, data, targets, transform=None):
		self.data = data
		self.targets = torch.FloatTensor(targets)
		self.transform = transform
		
	def __getitem__(self, index):
		x = self.data[index]
		y = self.targets[index]
		return x, y
	
	def __len__(self):
		return len(self.data)


def classifyRoi(roi, model):

	roi = cv2.resize(roi, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
	#convert to float
	roi =  roi / 255.0

	tensor_x = torch.Tensor([roi])
	signDataset = SignDataset(tensor_x,[1]) # create your datset
	dataLoader = DataLoader(signDataset) # create your dataloader
	for data in dataLoader:
		images, labels = data
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		return speedSigns[predicted]
	return ""

signCascade = cv2.CascadeClassifier('cascade/cascade.xml')

def detect(image, model):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	signs = signCascade.detectMultiScale(gray, 5, 5)
		
	roi = None

	for (x,y,w,h) in signs:
		roi = image[y:y+h, x:x+w]
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
		break

	if (roi is not None):
		speedText = classifyRoi(roi, model)

		cv2.rectangle(image, [x,y], [x+w, y+h], (255,0,255),4)
		cv2.putText(image, speedText, [x+2,y-5], cv2.FONT_HERSHEY_SIMPLEX, 
			1, (255,0,255), 2, cv2.LINE_AA)


	return image