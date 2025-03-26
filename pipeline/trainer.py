import torch
from torch.utils.data import DataLoader, random_split
from pipeline.dataset import CardDataset
from pipeline.model import CardClassifier
import torch.nn.functional as F


class Trainer:
    def __init__(self, num_epochs: int, is_13: bool, train_decks: list[str], test_decks: list[str]):
        # set training hyperparameters
        self.num_epochs = num_epochs

        # load dataset
        self.model = CardClassifier(is_13)
        print("Loading dataset...", end="", flush=True)
        train = CardDataset(decks=train_decks, is_13=is_13)
        test = CardDataset(decks=test_decks, is_13=is_13)
        print("done", flush=True)
        print(f"Train: {len(train)}    Test: {len(test)}")

        self.train_dl = DataLoader(train, batch_size=32, shuffle=True)
        self.test_dl = DataLoader(test, batch_size=32, shuffle=True)
        print(f"Train Batches: {len(self.train_dl)}    Test Batches: {len(self.test_dl)}")


    def eval_model(self):
        with torch.inference_mode():
            num_correct = 0
            num_seen = 0
            loss_sum = 0

            # iterate over testing data
            for test_imgs, test_labels in self.test_dl:
                test_pred: torch.FloatTensor = self.model(test_imgs)
                tmp_correct = (test_pred.argmax(dim=1) == test_labels).float().sum().item()
                num_correct += tmp_correct
                num_seen += len(test_labels)
                loss = F.cross_entropy(test_pred, test_labels)
                loss_sum += loss

        return loss_sum / len(self.test_dl), num_correct / num_seen


    def train(self):
        # Begin training
        optim = torch.optim.Adam(self.model.parameters(), lr=.001)
        for epoch in range(self.num_epochs):
            # train for one epoch
            train_loss = 0
            for imgs, labels in self.train_dl:
                pred = self.model(imgs)
                loss = F.cross_entropy(pred, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()
            train_loss /= len(self.train_dl)

            # test after each epoch
            test_loss, test_acc = self.eval_model()
            print(f"Epoch {epoch}, Train Loss: {train_loss:5.3f}, Test Loss: {test_loss:5.3f}, Test Acc: {test_acc:.3f}", flush=True)