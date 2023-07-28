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





classes =  {"bkg": 0, "sign":1}
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

        images=list(image.permute(2,0,1).to(device) for image in images)
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




















"""
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the model's input size (800x800)
    image = cv2.resize(image, (800, 800))
    # Convert the image to a PyTorch tensor and normalize to [0, 1]
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    # Add a batch dimension as the model expects a batch of images
    image_tensor = image_tensor.unsqueeze(0)

    # Move the image tensor to CUDA if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    return image_tensor




image_path = '../images/30.png'  

image_tensor = preprocess_image(image_path)
# Ensure the model is in evaluation mode
model.eval()

# Move the image tensor to CUDA if available
if torch.cuda.is_available():
    image_tensor = image_tensor.cuda()

# Perform inference
with torch.no_grad():
    outputs = model(image_tensor)
    print(outputs)
"""