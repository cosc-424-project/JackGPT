import torch
import torch.nn as nn
import torch.nn.functional as F


class CardClassifier(nn.Module):
    '''
    A simple CNN for classifing cards. Can classify the rank or the
    rank and suit as specified.
    '''
    def __init__(self, is_13: bool) -> None:
        super().__init__()
        self.is_13 = is_13
        self.conv1 = nn.Conv2d(1, 8, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 5)
        self.conv3 = nn.Conv2d(32, 128, 3)
        self.lin1 = nn.Linear(13312, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 13 if is_13 else 52)
    
    def forward(self, data: torch.Tensor, debug = False) -> torch.Tensor:
        '''
        Given a 90x126 image of a card, will forward propagate.

        Turning on `debug` will print the shape of each tensor along
        the way.
        '''
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
            tmp3 = self.pool(F.relu(self.conv3.forward(tmp2)))
            tmp4 = torch.flatten(tmp3, 1)
            tmp5 = F.relu(self.lin1.forward(tmp4))
            tmp6 = F.relu(self.lin2.forward(tmp5))
            tmp7 = self.lin3.forward(tmp6)
            return tmp7