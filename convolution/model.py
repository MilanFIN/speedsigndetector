import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 15, 5) #1,6,5 
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(15, 1, 5) #15,1,5
		self.fc1 = nn.Linear(1 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 9)
	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		x = F.log_softmax(x, dim=1)
		return x
	



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
