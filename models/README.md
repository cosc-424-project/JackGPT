# Models

A few Python files for loading the card dataset, creating the
model class, and training the model. Both the 13-class and
52-class models are trained in `trainer.py`.

## How to Use

First, create a new directory which has the following
architecture:

```
models
    processed
        ace_of_clubs
        ace_of_diamonds
        ...
        king_of_spades
```

The 50 images in each of the subdirectories must be grayscale
and have dimensions 126x90.

After creating this folder, run the following command:
```
python trainer.py
```
This will first train the 13-class model, then the 52-class model.