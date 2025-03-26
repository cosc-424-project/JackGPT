from pipeline.datagen import datagen
from pipeline.trainer import Trainer
import os

# ensure dataset is generated
datagen()

# create and train model
decks = os.listdir("processed")
for deck in decks:
    print(f"\nTesting {deck}")
    testing_deck = deck
    training_decks = decks.copy()
    training_decks.remove(deck)

    t13 = Trainer(
        num_epochs=5,
        is_13=True,
        train_decks=training_decks,
        test_decks=[testing_deck]
    )
    t13.train()