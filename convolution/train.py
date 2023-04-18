import cv2
import numpy as np
import pickle
import json
import random
from skimage.util import random_noise

from model import Net, SignDataset
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
saveName = "signModel"

random.seed(1001)



def getDataset(iterations):
	outputs = []
	inputs = []


	iteration = 0
	while iteration < iterations:
		for i, sign in enumerate(speedSigns):


			signImg = cv2.imread("sourceImages/"+sign+".jpg")
			graySign = cv2.cvtColor(signImg, cv2.COLOR_BGR2GRAY) #


			xPerChange = random.randint(-30, 30)
			yPerChange = random.randint(-30, 30)
			num_rows, num_cols = graySign.shape[:2] 

			topLeft = [0,0]
			botLeft = [0,num_rows]
			topRight = [num_cols, 0]
			botRight = [num_cols, num_rows]
			
			topLeft[1] += xPerChange
			botLeft[1] -= xPerChange
			topRight[1] -= xPerChange
			botRight[1] += xPerChange

			topLeft[0] += yPerChange
			topRight[0] -= yPerChange

			botLeft[0] -= yPerChange
			botRight[0] += yPerChange

			pts1 = np.float32([topLeft, botLeft, botRight, topRight])
			pts2 = np.float32([[0,0], [0,num_rows], [num_cols, num_rows], [num_cols, 0]])
			M = cv2.getPerspectiveTransform(pts1, pts2)
			graySign = cv2.warpPerspective(graySign, M, (num_cols, num_rows),
													borderMode=cv2.BORDER_CONSTANT,
													borderValue=(255,255,255))


			origY, origX = graySign.shape[:2] 
			graySign = cv2.copyMakeBorder(graySign,100, 100, 100, 100,cv2.BORDER_CONSTANT,value=(255,255,255))


			#rotation
			angle = 0#random.randint(-10, 10)
			scale = random.uniform(0.5, 1.5)
			(h, w) = graySign.shape[:2]
			center = (w // 2, h // 2)
			M = cv2.getRotationMatrix2D(center, angle,scale)

			graySign = cv2.warpAffine(graySign, M, (w, h),
									borderMode=cv2.BORDER_CONSTANT,
									borderValue=(255,255,255))
						




			xTranslation = random.randint(-50, 50)
			yTranslation = random.randint(-50, 50)


			translation_matrix = np.float32([ [1,0,xTranslation], [0,1,yTranslation] ])   
			num_rows, num_cols = graySign.shape[:2] 

			graySign = cv2.warpAffine(graySign, translation_matrix, (num_cols, num_rows),
									borderMode=cv2.BORDER_CONSTANT,
									borderValue=(255,255,255))
			
			graySign = graySign[100:100+origY, 100:100+origX]




			minNoise = 0.01
			maxNoise = 0.2
			noiseCount = random.uniform(minNoise, maxNoise)
			graySign = random_noise(graySign, mode='s&p', amount=noiseCount)


			#back to 8 bit
			#graySign = np.array(255 * graySign, dtype=np.uint8)

			#size = random.randint(30, 50)
			#graySign = cv2.resize(graySign, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

			graySign = cv2.resize(graySign, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)



			outputs.append(i)
			inputs.append(graySign) #.flatten()
			"""
			"""
			cv2.imshow("test", graySign)
			cv2.waitKey(300)

		iteration += 1
		print("making training data: ", iteration, " / ", iterations)


	outputs = np.eye(len(speedSigns))[outputs]
	inputs = np.array(inputs)
	
	return inputs, outputs
	




inputs, outputs = getDataset(1000)


tensor_x = torch.Tensor(inputs) # transform to torch tensor
tensor_y = torch.Tensor(outputs)
signDataset = SignDataset(tensor_x,tensor_y) # create your datset
dataLoader = DataLoader(signDataset) # create your dataloader


#Initialize the model       
net = Net()


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.1)
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.1)


for epoch in range(100):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(dataLoader, 0):
		signs, labels = data

		optimizer.zero_grad()

		predictions = net(signs)
		loss = criterion(predictions, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		#if i % 500 == 499:   
	print(f'[{epoch + 1}, {i + 1:5d}] loss: {(running_loss/i):.5f}')
	running_loss = 0.0

print('Finished Training')



torch.save(net.state_dict(), "./models/"+saveName+".pt")


inputs, outputs = getDataset(10)
tensor_x = torch.Tensor(inputs)
tensor_y = torch.Tensor(outputs)
my_dataset = SignDataset(tensor_x,tensor_y)
my_dataloader = DataLoader(my_dataset)



correct = 0
total = 0
with torch.no_grad():
	for data in my_dataloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
