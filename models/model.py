'''
64
16
256
256
13
'''

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CardDataset13

class M13 (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 11)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 7)
        self.lin1 = nn.Linear(3536, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 13)
    
    def forward(self, data: torch.FloatTensor, debug = False):
        if debug:
            print(data.shape)
            tmp1 = self.pool(F.relu(self.conv1.forward(data)))
            print(tmp1.shape)
            tmp2 = self.pool(F.relu(self.conv2.forward(tmp1)))
            print(tmp2.shape)
            tmp3 = torch.flatten(tmp2, 1)
            print(tmp3.shape)
            tmp4 = F.relu(self.lin1.forward(tmp3))
            print(tmp4.shape)
            tmp5 = F.relu(self.lin2.forward(tmp4))
            print(tmp5.shape)
            tmp6 = F.relu(self.lin3.forward(tmp5))
            print(tmp6.shape)
            return tmp6
        else:
            tmp1 = self.pool(F.relu(self.conv1.forward(data)))
            tmp2 = self.pool(F.relu(self.conv2.forward(tmp1)))
            tmp3 = torch.flatten(tmp2, 1)
            tmp4 = F.relu(self.lin1.forward(tmp3))
            tmp5 = F.relu(self.lin2.forward(tmp4))
            tmp6 = F.relu(self.lin3.forward(tmp5))
            return tmp6


data = torch.rand((1, 32, 32))
m13 = M13()
dataset = CardDataset13()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(m13.parameters())

NUM_EPOCHS = 30

ctr = 0
running_loss = 0
for epoch in range(NUM_EPOCHS):
    for imgs, labels in dataloader:
        # train
        pred = m13(imgs)
        loss: torch.FloatTensor = loss_fn(pred, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        # test if necessary
        if ctr % 100 == 0:
            print(f"{ctr:<8d}: {loss / 100:8.3f}", flush=True)
            running_loss = 0
        ctr += 1