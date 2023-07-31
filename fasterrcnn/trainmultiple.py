import torch
import torch.nn as nn
import torchvision
import cv2
from dataset import CustomDataset, Compose
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import transforms as T
import numpy as np


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))







classes = {
    0: "bkg",
	1: "10_SIGN",
	2: "20_SIGN",
	3: "30_SIGN",
	4: "40_SIGN",
	5: "50_SIGN",
	6: "60_SIGN",
	7: "70_SIGN",
	8: "80_SIGN",
	9: "90_SIGN",
	10: "100_SIGN",
	11: "110_SIGN",
	12: "120_SIGN",
}



num_classes = len(classes)
dataset = CustomDataset("data/images", "data/annotations", classes, get_transform(train=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move the model to CUDA if available
if torch.cuda.is_available():
	model = model.cuda()


#Set up data loader
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

#Loss function
criterion = torch.nn.CrossEntropyLoss() 

#Optimization
optimizer = torch.optim.SGD(model.parameters(), lr=0.002,
                                momentum=0.9, weight_decay=0.0005)

#Training loop
model.train()
num_epochs = 10
for epoch in range(num_epochs):

    totalLosses = []
    for i, (images, targets) in enumerate(train_loader):

        images=list(image.to(device) for image in images)
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
              # Forward pass
        lossDict=model(images,targets)
        losses=sum(loss for loss in lossDict.values())
        lossesValue=losses.item()
        totalLosses.append(lossesValue)
        print("epoch: "+ str(epoch) +" batch: " + str(i) + " loss: " + str(lossesValue))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print("epoch: " + str(epoch) + " mean loss: " + str(np.mean(totalLosses)))
    torch.save(model.state_dict(), "./models/" + str(epoch) + ".pt")

















