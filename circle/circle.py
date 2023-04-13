import cv2
import numpy as np
import pytesseract


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

def getTextForRoi(roi):
	redRoi = roi.copy()
	redRoi[:,:,0] = np.zeros([redRoi.shape[0], redRoi.shape[1]])
	redRoi[:,:,1] = np.zeros([redRoi.shape[0], redRoi.shape[1]])

	grayRoi = cv2.cvtColor(redRoi, cv2.COLOR_BGR2GRAY) 
	roiMax = np.max(grayRoi)

	ret, threshedRoi = cv2.threshold(grayRoi, roiMax *0.5, 255, cv2.THRESH_BINARY_INV)

	roiKSize = (3, 3)

	blurredRoi = cv2.blur(threshedRoi, roiKSize)

	inverseGrayRoi = cv2.bitwise_not(grayRoi)
	finalRoi = cv2.bitwise_and(inverseGrayRoi, inverseGrayRoi, mask=blurredRoi)
	
	kernel = np.zeros((3,3),np.uint8)
	erosion = cv2.erode(finalRoi,kernel,iterations = 1)
	dilate = cv2.dilate(erosion,kernel,iterations = 1)

	rawText = pytesseract.image_to_string(dilate, config="--psm 11 -c tessedit_char_whitelist=0123456789")#
	speedText = ""
	for s in rawText:
		if (s.isdigit()):
			speedText += s
			
	return speedText



def detect(image, model=None):

	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray, 5)

	minDist = 100
	param1 = 50 #500
	param2 = 75 #200 #smaller value-> more false circles
	minRadius = 18
	maxRadius = 100 #10
	circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 0.8, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

	centerX = 0
	centerY = 0
	radius = 0
	foundSign = False
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			#cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
			centerX = i[0]
			centerY = i[1]
			radius = i[2]
			foundSign = True
			break
	
	if (not foundSign): # or centerX + radius > image.shape[1] or centerY + radius > image.shape[0]     		or centerX - radius < 0 or centerY - radius < 0
		return image
	x = centerX - radius
	y = centerY - radius
	w = 2*radius
	h = 2*radius

	roi = image[y:y+h, x:x+w]

	if (len(roi) ==0):
		return image

	if (model is None):
		speedText = getTextForRoi(roi)
	else:
		speedText = classifyRoi(roi, model)

	cv2.rectangle(image, [x,y], [x+w, y+h], (255,0,255),4)
	cv2.putText(image, speedText, [x+2,y-5], cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,255), 2, cv2.LINE_AA)

	return image