from pipeline.datagen import datagen
from pipeline.trainer import Trainer
from sklearn.metrics import ConfusionMatrixDisplay
from pipeline.helpers import CARD_VALS, CARD_SUITS
import matplotlib.pyplot as plt
import os

# ensure dataset is generated
datagen()


# create results-storing dictionary and list
results = {
    "train_losses": {},
    "test_losses": {},
    "test_accs": {},
    "display_images": {},
    "display_preds": {},
    "display_true_labels": {},
}
confusions = []


# create and train model
NUM_EPOCHS = 5
decks = os.listdir("processed")
for deck in decks:
    print(f"\nUsing {deck} as testing dataset")
    testing_deck = deck
    training_decks = decks.copy()
    training_decks.remove(deck)

    t13 = Trainer(
        num_epochs=NUM_EPOCHS,
        is_13=False,
        train_decks=training_decks,
        test_decks=[testing_deck]
    )
    train_losses, test_losses, test_accs, confusion = t13.train()

    # add to results dict
    results["train_losses"][deck] = train_losses
    results["test_losses"][deck] = test_losses
    results["test_accs"][deck] = test_accs
    confusions.append(confusion)

    image, pred_label, true_label = t13.display_samples()
    results["display_images"][deck] = image
    results["display_preds"][deck] = pred_label
    results["display_true_labels"][deck] = true_label

for deck in decks:
    # cv2.namedWindow(f"{deck}: Predicted = {results['display_preds'][deck]}, True = {results['display_true_labels'][deck]}", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(f"{deck}: Predicted = {results['display_preds'][deck]}, True = {results['display_true_labels'][deck]}", width=450, height=635)
    # cv2.imshow(f"{deck}: Predicted = {results['display_preds'][deck]}, True = {results['display_true_labels'][deck]}", results["display_images"][deck])
    # cv2.waitKey()
    plt.imshow(results["display_images"][deck], cmap='gray')
    plt.title(f"{deck}: Predicted = {results['display_preds'][deck]}, True = {results['display_true_labels'][deck]}")
    plt.axis('off')
    plt.show()

# display confusion matrices
for i in range(len(decks)):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusions[i])
    disp.plot()
    plt.title(decks[i])
    plt.show()

# graph results
CATEGORIES = [
    "train_losses",
    "test_losses",
    "test_accs"
]
TITLES = [
    "Training Loss vs. Epochs",
    "Testing Loss vs. Epochs",
    "Test Accuracy vs. Epochs"
]
Y_LABELS = [
    "Loss", "Loss", "Accuracy"
]

EPOCHS = range(NUM_EPOCHS)
for i in range(len(CATEGORIES)):
    plt.figure(figsize=(10, 6))
    plt.title(TITLES[i])
    plt.xlabel("Epochs")
    plt.ylabel(Y_LABELS[i])
    for deck in decks:
        plt.plot(EPOCHS, results[CATEGORIES[i]][deck], label=deck)
    plt.legend()
    plt.show()
