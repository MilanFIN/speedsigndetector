import torch
import torch.nn as nn
import torchvision
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import transforms as T
import numpy as np

device = "cpu"
classes =  {0:"bkg", 1:"sign"}


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
num_classes = len(classes)

in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


# Move the model to CUDA if available
if torch.cuda.is_available():
	model = model.cuda()
	device = "cuda"


model.load_state_dict(torch.load("./fasterrcnn/models/single.pt", map_location=torch.device(device) ))
model.eval()


def overlap(box1, box2):
    # Check if two bounding boxes overlap
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    return not (x1_1 < x0_2 or x0_1 > x1_2 or y1_1 < y0_2 or y0_1 > y1_2)



def detect(image):
	tensorImage =torchvision.transforms.ToTensor()(image)

	tensorImage.permute(2,0,1).to(device)

	result = model([tensorImage])

	boxes = result[0]["boxes"]
	labels = result[0]["labels"]
	scores = result[0]["scores"]

	validPredictions = []

	for (i, box) in enumerate(boxes):
		label = labels[i].detach().item()
		score = scores[i].detach().item()

		if (score > 0.2):

			borders = box.detach().numpy()
			x0 = int(borders[0])
			y0 = int(borders[1])
			x1 = int(borders[2])
			y1 = int(borders[3])

			valid = True

			for validBox in validPredictions:
				if (overlap((x0,y0,x1,y1), validBox)):
					valid = False
					break

			if (valid):
				cv2.rectangle(image, (x0,y0), (x1,y1), (255,255,0), 2)

				cv2.putText(image, classes[label] + ": " + str(int(score*100)) + "%", [x0+2,y0-5], cv2.FONT_HERSHEY_SIMPLEX, 
					1, (255,255,0), 2, cv2.LINE_AA)
				validPredictions.append((x0, y0, x1, y1))

	return image
