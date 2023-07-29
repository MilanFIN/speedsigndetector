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

#image = cv2.imread("images/30.png")


def detect(image):
	tensorImage =torchvision.transforms.ToTensor()(image)

	tensorImage.permute(2,0,1).to(device)

	result = model([tensorImage])

	boxes = result[0]["boxes"]
	labels = result[0]["labels"]
	scores = result[0]["scores"]

	for (i, box) in enumerate(boxes):
		label = labels[i].detach().item()
		score = scores[i].detach().item()

		if (score > 0.5):

			borders = box.detach().numpy()
			x0 = int(borders[0])
			y0 = int(borders[1])
			x1 = int(borders[2])
			y1 = int(borders[3])

			cv2.rectangle(image, (x0,y0), (x1,y1), (255,255,0), 2)

			print(type(label), label, classes)
			cv2.putText(image, classes[label] + ": " + str(int(score*100)) + "%", [x0+2,y0-5], cv2.FONT_HERSHEY_SIMPLEX, 
				1, (255,255,0), 2, cv2.LINE_AA)

	return image

	#cv2.imshow("test", image)
	#cv2.waitKey(3000)