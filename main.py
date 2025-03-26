from pipeline.datagen import datagen
from pipeline.trainer import Trainer
import os

# ensure dataset is generated
datagen()

# create and train model
decks = os.listdir("processed")
t13 = Trainer(
    num_epochs=10,
    is_13=False,
    train_decks=decks[0:-1],
    test_decks=[decks[-1]]
)
t13.train()