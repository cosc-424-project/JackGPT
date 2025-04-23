import torch
import torch.nn as nn
import torch.nn.functional as F


class CardClassifier(nn.Module):
    '''
    A simple CNN for classifing cards. Can classify the rank or the
    rank and suit as specified. Set `dropout=0` to avoid dropout regularization.
    '''
    def __init__(self, num_classes: bool, do_bnorm: bool, dropout: float) -> None:
        super().__init__()
        self.is_13 = num_classes
        self.conv1 = nn.Conv2d(1, 8, 7)
        self.bnorm1 = nn.BatchNorm2d(8) if do_bnorm else self.identity
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 5)
        self.bnorm2 = nn.BatchNorm2d(32) if do_bnorm else self.identity
        self.conv3 = nn.Conv2d(32, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128) if do_bnorm else self.identity
        self.lin1 = nn.Linear(13312, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, num_classes)

    def identity(self, data):
        return data
    
    def forward(self, data: torch.Tensor, debug = False) -> torch.Tensor:
        '''
        Given a 90x126 image of a card, will forward propagate.

        Turning on `debug` will print the shape of each tensor along
        the way.
        '''
        tmp1 = self.pool(F.relu(self.dropout(self.bnorm1(self.conv1.forward(data)))))
        tmp2 = self.pool(F.relu(self.dropout(self.bnorm2(self.conv2.forward(tmp1)))))
        tmp3 = self.pool(F.relu(self.dropout(self.bnorm3(self.conv3.forward(tmp2)))))
        tmp4 = torch.flatten(tmp3, 1)
        tmp5 = F.relu(self.lin1.forward(tmp4))
        tmp6 = F.relu(self.lin2.forward(tmp5))
        tmp7 = self.lin3.forward(tmp6)
        return tmp7