from pipeline.datagen import datagen
from pipeline.trainer import Trainer

# ensure dataset is generated
datagen()

# create and train model
t13 = Trainer(5, True)
t13.train()