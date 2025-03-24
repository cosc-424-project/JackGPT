import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
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
            tmp6 = self.lin3.forward(tmp5)
            return tmp6


# Create model
m13 = M13()

# Curate data
TRAIN_FRAC = .8
NUM_EPOCHS = 30

data = torch.rand((1, 32, 32))
dataset = CardDataset13()
train, test = random_split(dataset, [TRAIN_FRAC, 1 - TRAIN_FRAC])
print(f"Train: {len(train)}    Test: {len(test)}")

train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=32, shuffle=True)
print(f"Train Batches: {len(train_dl)}    Test Batches: {len(test_dl)}")


def eval_model():
    with torch.inference_mode():
        num_correct = 0
        num_seen = 0
        loss_sum = 0

        # iterate over testing data
        for test_imgs, test_labels in test_dl:
            test_pred: torch.FloatTensor = m13(test_imgs)
            tmp_correct = (test_pred.argmax(dim=1) == test_labels).float().sum().item()
            num_correct += tmp_correct
            num_seen += len(test_labels)
            loss = F.cross_entropy(test_pred, test_labels)
            loss_sum += loss

    return loss_sum / len(test_dl), num_correct / num_seen


# Begin training
optim = torch.optim.Adam(m13.parameters(), lr=.001)
for epoch in range(NUM_EPOCHS):
    # train for one epoch
    train_loss = 0
    for imgs, labels in train_dl:
        pred = m13(imgs)
        loss = F.cross_entropy(pred, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)

    # test after each epoch
    test_loss, test_acc = eval_model()
    print(f"Epoch {epoch}, Train Loss: {train_loss:5.3f}, Test Loss: {test_loss:5.3f}, Test Acc: {test_acc:.3f}", flush=True)