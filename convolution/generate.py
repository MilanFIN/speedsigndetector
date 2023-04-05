import cv2
import numpy as np
import pickle
import json
import random
from skimage.util import random_noise

from model import Net
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

random.seed(1001)



def getDataset(iterations):
	outputs = []
	inputs = []


	iteration = 0
	while iteration < iterations:
		for i, sign in enumerate(speedSigns):


			signImg = cv2.imread("sourceImages/"+sign+".jpg")
			graySign = cv2.cvtColor(signImg, cv2.COLOR_BGR2GRAY) #

			
			num_rows, num_cols = graySign.shape[:2] 

			xTranslation = random.randint(-20, 20)
			yTranslation = random.randint(-20, 20)


			translation_matrix = np.float32([ [1,0,xTranslation], [0,1,yTranslation] ])   

			graySign = cv2.warpAffine(graySign, translation_matrix, (num_cols, num_rows),
									borderMode=cv2.BORDER_CONSTANT,
									borderValue=(255,255,255))
			


			xPerChange = random.randint(-30, 30)
			yPerChange = random.randint(-30, 30)

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


			minNoise = 0.01
			maxNoise = 0.2
			noiseCount = random.uniform(minNoise, maxNoise)
			graySign = random_noise(graySign, mode='s&p', amount=noiseCount)

			#back to 8 bit
			graySign = np.array(255 * graySign, dtype=np.uint8)

			size = random.randint(30, 50)
			graySign = cv2.resize(graySign, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

			graySign = cv2.resize(graySign, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)


			cv2.imshow("test", graySign)

			outputs.append(i)
			inputs.append(graySign) #.flatten()


			cv2.waitKey(1)

		iteration += 1

	outputs = np.eye(len(speedSigns))[outputs]
	inputs = np.array(inputs)
	return inputs, outputs
	




class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.FloatTensor(targets)
        print(len(self.targets), len(self.data))
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y
    
    def __len__(self):
        return len(self.data)


#X = torch.from_numpy(inputs).type(torch.FloatTensor)
#y = torch.from_numpy(outputs).type(torch.FloatTensor)

inputs, outputs = getDataset(20)

tensor_x = torch.Tensor(inputs) # transform to torch tensor
tensor_y = torch.Tensor(outputs)
my_dataset = MyDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset) # create your dataloader


#Initialize the model       
net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.1)


for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(my_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #if i % 2000 == 1999:    # print every 2000 mini-batches
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    running_loss = 0.0

print('Finished Training')




inputs, outputs = getDataset(20)
tensor_x = torch.Tensor(inputs) # transform to torch tensor
tensor_y = torch.Tensor(outputs)
my_dataset = MyDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset) # create your dataloader


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in my_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
