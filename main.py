import re
import cv2
import numpy as np
import time
from os import listdir
from os.path import isfile, join
import pytesseract

from redChannel import redChannel
from circle import circle
from sift import sift
from brisk import brisk

import sys


channel_initials = list('BGR')
windowName = "drive"

algo = "color"

model = None

if (algo == "color"):
	from convolution.model import Net, SignDataset
	import torch
	import torch.optim as optim
	from torchvision import transforms
	from torch.utils.data import Dataset, DataLoader

	import torch.nn as nn
	import torch.nn.functional as F

	model = Net()
	
	model.load_state_dict(torch.load("./convolution/models/signModel.pt"))






files = [f for f in listdir("images/") if isfile(join("images/", f))]

videoFolder = "videos/"
videoName = "Driving in Finland Short Drive in Tampere, Finland.mp4"

cap = cv2.VideoCapture(videoFolder+videoName)
cap.set(cv2.CAP_PROP_POS_FRAMES, 5200)#5200
count = 0
elapsedTime = 0
fps = 10
while cap.isOpened():
	start = time.time()
	ret,frame = cap.read()
	frame = redChannel.detect(frame, model)
	cv2.imshow(windowName, frame)
	count = count + 1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	elapsedTime = (time.time() - start)
	print(count)


cap.release()
cv2.destroyAllWindows() # destroy all opened windows
"""

for file in reversed(files):
	img = cv2.imread("images/"+file)
	#img = parseColor(img)
	img = redChannel.detect(img, model)
	cv2.imshow(windowName, img)

	if cv2.waitKey(3000) & 0xFF == ord('q'):
		break

"""
