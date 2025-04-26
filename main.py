from pipeline.datagen import datagen
from pipeline.trainer import Trainer
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


# ensure dataset is generated
# datagen()


# create results-storing dictionary and list
def pipeline(
    num_epochs: int,
    num_classes: int,
    do_augment: bool,
    do_bnorm: bool,
    dropout: float,
    do_show: bool,
    output_dir: str | None,
) -> None:
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
    decks = os.listdir("processed")
    for deck in decks:
        print(f"\nUsing {deck} as testing dataset")
        testing_deck = deck
        training_decks = decks.copy()
        training_decks.remove(deck)

        trainer = Trainer(
            num_epochs=num_epochs,
            num_classes=num_classes,
            train_decks=training_decks,
            test_decks=[testing_deck],
            do_augment=do_augment,
            do_bnorm=do_bnorm,
            dropout=dropout
        )
        train_losses, test_losses, test_accs, confusion = trainer.train()
        trainer.save(f"ignore_data/models/test_{deck}+epoch_3.pt")

        # add to results dict
        results["train_losses"][deck] = train_losses
        results["test_losses"][deck] = test_losses
        results["test_accs"][deck] = test_accs
        confusions.append(confusion)

        image, pred_label, true_label = trainer.display_samples()
        results["display_images"][deck] = image
        results["display_preds"][deck] = pred_label
        results["display_true_labels"][deck] = true_label

    # create output path if not exist
    if output_dir != None and output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # show an image from each deck, along with the true and predicted labels
    plt.ioff()
    for deck in decks:
        plt.figure()
        plt.imshow(results["display_images"][deck], cmap='gray')
        plt.title(f"{deck}: Predicted = {results['display_preds'][deck]}, True = {results['display_true_labels'][deck]}")
        plt.axis('off')
        if output_dir != None and output_dir != "":
            plt.savefig(output_dir + f'/example_{deck}.png')
        if do_show:
            plt.show()
        plt.close()

    # display confusion matrices
    for i in range(len(decks)):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusions[i])
        disp.plot(text_kw={"fontsize": 5})

        plt.xticks(rotation=-45, fontsize=5) 
        plt.yticks(fontsize=5)
        plt.title(decks[i])
        if output_dir != None and output_dir != "":
            plt.savefig(output_dir + f'/cm_{decks[i]}.png')
        if do_show:
            plt.show()
        plt.close()

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

    EPOCHS = range(num_epochs)
    for i in range(len(CATEGORIES)):
        plt.figure(figsize=(10, 6))
        plt.title(TITLES[i])
        plt.xlabel("Epochs")
        plt.ylabel(Y_LABELS[i])
        for deck in decks:
            plt.plot(EPOCHS, results[CATEGORIES[i]][deck], label=deck)
        plt.legend()
        if output_dir != None and output_dir != "":
            plt.savefig(output_dir + f'/{CATEGORIES[i]}.png')
        if do_show:
            plt.show()
        plt.close()

    acc_ctr = 0
    final_accs = []
    for deck in decks:
        final_accs.append(results["test_accs"][deck][-1])
        acc_ctr += results["test_accs"][deck][-1]
    
    if do_show:
        print(f"\nFinal accuracy: {100 * acc_ctr / len(decks):.2f}%")
    if output_dir != None and output_dir != "":
        f = open(output_dir + '/accuracy.txt', 'w')
        for i in range(len(decks)):
            f.write(f"{decks[i]}: {final_accs[i]:.4f}%\n")
        f.write(f"\nFinal accuracy: {100 * acc_ctr / len(decks):.2f}%")
        f.close()


# # actually run the pipeline
# pipeline(
#     num_epochs=5,
#     num_classes=10,
#     do_augment=True,
#     do_bnorm=True,
#     dropout=.2,
#     output_dir="results/test",
#     do_show=False
# )


# hyperparameter optimizations
'''
batch norm?
data aug?
dropout?
10, 13, 52 classes?

_bn_da_13
'''

BOOL_VALS = [True, False]
CLASS_NUMS = [10, 13, 52]

for do_bnorm in BOOL_VALS:
    for do_augment in BOOL_VALS:
        for do_dropout in BOOL_VALS:
            for num_classes in CLASS_NUMS:
                print(f"\nWorking on results/{'_bn' if do_bnorm else ''}{'_da' if do_augment else ''}{'_do' if do_dropout else ''}_{num_classes}")
                pipeline(
                    num_epochs=5,
                    num_classes=num_classes,
                    do_augment=do_augment,
                    do_bnorm=do_bnorm,
                    dropout=.2 if do_dropout else .0,
                    output_dir=f"results/{'_bn' if do_bnorm else ''}{'_da' if do_augment else ''}{'_do' if do_dropout else ''}_{num_classes}",
                    do_show=False
                )