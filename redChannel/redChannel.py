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
	print(roi)
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



def detect(image, model = None):
	ksize = (3, 3)

	result = image.copy()

	#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	#image = cv2.blur(image, ksize)
	boundaries = [([0, 0, 150], [150, 150, 255])]
	#[([0, 0, 100], [80, 80, 255])]

	(lower, upper) = boundaries[0]
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	mask = cv2.inRange(image, lower, upper)
	result = cv2.bitwise_and(image, image, mask=mask)

	#cv2.imshow('mask', mask)
	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) #
	gray = cv2.blur(gray, ksize)

	avg = np.average(gray)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

	contours, hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	foundContour = False
	for cnt in contours:
		if 200<cv2.contourArea(cnt)<5000:
			x,y,w,h = cv2.boundingRect(cnt)
			#speed signs are relatively round, so ignoring results with too much variation
			if (w/h < 1.2 and h/w < 1.2):
				contour = cnt
				foundContour = True
				#cv2.drawContours(image,[cnt],0,(0,255,0),2)
				#cv2.drawContours(mask,[cnt],0,255,-1)
				break
		
	if (not foundContour):
		return image

	x,y,w,h = cv2.boundingRect(contour)
	roi = image[y:y+h, x:x+w]

	if (model != None):
		speedText = classifyRoi(roi, model);
	else:
		speedText = getTextForRoi(roi)

	cv2.rectangle(image, [x,y], [x+w, y+h], (255,0,255),4)
	cv2.putText(image, speedText, [x+2,y-5], cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,255), 2, cv2.LINE_AA)
	#cv2.imshow('result', image)
	#cv2.waitKey(3000)
	return image