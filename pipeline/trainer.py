import numpy as np
import torch
from torch.utils.data import DataLoader
from pipeline.dataset import CardDataset
from pipeline.model import CardClassifier
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from pipeline.helpers import CARD_VALS, CARD_SUITS


class Trainer:
    '''
    A glorified training-testing loop for the pipeline.model.CardClassifier. Trains the model on
    `train_decks` and tests the model on `test_decks`. Specify whether to use 13 or 52 classes for
    the training.
    '''
    def __init__(self, num_epochs: int, num_classes: int, train_decks: list[str], test_decks: list[str], do_augment: bool, do_bnorm: bool, dropout: float) -> None:
        # set training hyperparameters
        self.num_epochs = num_epochs
        self.num_classes = num_classes

        # load dataset
        self.model = CardClassifier(num_classes, do_bnorm, dropout)
        print("Loading dataset...", end="", flush=True)
        train = CardDataset(decks=train_decks, num_classes=num_classes, do_augment=do_augment)
        test = CardDataset(decks=test_decks, num_classes=num_classes, do_augment=do_augment)
        print("done", flush=True)
        print(f"Train: {len(train)}    Test: {len(test)}")

        self.train_dl = DataLoader(train, batch_size=32, shuffle=True)
        self.test_dl = DataLoader(test, batch_size=32, shuffle=True)
        print(f"Train Batches: {len(self.train_dl)}    Test Batches: {len(self.test_dl)}")

        self.display = DataLoader(test, batch_size=1, shuffle=True)

    def eval_model(self) -> tuple[float, float, list[int], list[float]]:
        '''
        Using the model's existing parameters, returns the average loss, accuracy,
        true labels, and predicted labels.
        '''
        with torch.inference_mode():
            self.model.eval()
            num_correct = 0
            num_seen = 0
            loss_sum = 0

            true_labels = []
            pred_labels = []

            # iterate over testing data
            for test_imgs, test_labels in self.test_dl:
                test_pred: torch.FloatTensor = self.model(test_imgs)
                final_pred = test_pred.argmax(dim=1)
                tmp_correct = (final_pred == test_labels).float().sum().item()
                num_correct += tmp_correct
                num_seen += len(test_labels)
                loss = F.cross_entropy(test_pred, test_labels)
                loss_sum += loss

                true_labels += test_labels
                pred_labels += final_pred
            
            self.model.train()

        return loss_sum / len(self.test_dl), num_correct / num_seen, true_labels, pred_labels


    def train(self) -> tuple[list[float], list[float], list[float], np.ndarray]:
        '''
        Trains the model using the training deck, returning the training loss, test losses, test
        accuracies, and test confusion matrix.
        '''
        # Begin training
        train_losses = []
        test_losses = []
        test_accs = []

        true_labels: list[float]
        pred_labels: list[float]

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
            test_loss, test_acc, true_labels, pred_labels = self.eval_model()

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            print(f"Epoch {epoch}, Train Loss: {train_loss:5.3f}, Test Loss: {test_loss:5.3f}, Test Acc: {test_acc:.3f}", flush=True)
        
        return train_losses, test_losses, test_accs, confusion_matrix(true_labels, pred_labels)
    
    def display_samples(self) -> tuple[np.ndarray, str, str]:
        '''
        Returns the NumPy array of an image in the test dataset, along with the true and predicted labels.
        '''
        for batch in self.display:
            display_image, display_label = batch
            
            display_pred : torch.FloatTensor = self.model(display_image)
            display_pred_label = display_pred.argmax(dim=1)
            
            image_np = display_image.squeeze(0).permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)

            pred_str = str(display_pred_label.item())
            true_str = str(display_label.item())
            if self.num_classes == 13:
                pred_str = CARD_VALS[display_pred_label % 13]
                true_str = CARD_VALS[display_label % 13]
            elif self.num_classes == 52:
                pred_str = f"{CARD_VALS[display_pred_label // 4]}_of_{CARD_SUITS[display_pred_label % 4]}"
                true_str = f"{CARD_VALS[display_label // 4]}_of_{CARD_SUITS[display_label % 4]}"

            return image_np, pred_str, true_str
        
    def save(self, path: str) -> None:
        torch.save(self.model, path)