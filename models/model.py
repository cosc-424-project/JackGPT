import torch
import torch.nn as nn
import torch.nn.functional as F


class CardClassifier(nn.Module):
    # assumes 126x90 card images
    def __init__(self, is_13):
        super().__init__()
        self.is_13 = is_13
        self.conv1 = nn.Conv2d(1, 4, 11)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 7)
        self.lin1 = nn.Linear(3536, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 13 if is_13 else 52)
    
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
            tmp6 = self.lin3.forward(tmp5)
            return tmp6