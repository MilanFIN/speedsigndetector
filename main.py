import re
import cv2
import numpy as np
import time
from os import listdir
from os.path import isfile, join
import pytesseract


import sys
import argparse



def detect(method, secondary=None):
	windowName = "drive"

	files = [f for f in listdir("images/") if isfile(join("images/", f))]

	for file in reversed(files):
		img = cv2.imread("images/"+file)
		#img = parseColor(img)
		if (secondary is not None):
			img = method.detect(img, secondary)
		else:
			img = method.detect(img)
		cv2.imshow(windowName, img)

		if cv2.waitKey(3000) & 0xFF == ord('q'):
			break



parser = argparse.ArgumentParser("")
parser.add_argument("--algo", help="select algorithm", type=str)
parser.add_argument("--classifier", help="optional parameter to specify usage of a neural network", type=str)

args = vars(parser.parse_args())

model = None

if (args["classifier"] == "cnn" ):
	from convolution.model import Net, SignDataset
	import torch
	import torch.optim as optim
	from torchvision import transforms
	from torch.utils.data import Dataset, DataLoader

	import torch.nn as nn
	import torch.nn.functional as F

	model = Net()
	
	model.load_state_dict(torch.load("./convolution/models/signModel.pt"))


if (args["algo"] is not None):
	algo = args["algo"]

	if (algo == "color"):
		from color import color
		detect(color, model)
	elif (algo == "shape"):
		from circle import circle
		detect(circle, model)
	elif (algo == "sift"):
		from sift import sift
		detect(sift)
	elif (algo == "brisk"):
		from brisk import brisk
		detect(brisk)
	elif (algo == "haar"):
		from cascade import cascade
		from convolution.model import Net
		import torch
		import torch.optim as optim
		import torch.nn as nn
		import torch.nn.functional as F

		model = Net()
		model.load_state_dict(torch.load("./convolution/models/signModel.pt"))

		detect(cascade, model)
else:
	print("missing -algo parameter")

sys.exit(0)



algo = "color"

model = None







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
