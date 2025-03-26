# JackGPT

A card-recognition model and card dataset creation pipeline geared towards blackjack. The model supports a 13-class option, since suits do not affect gameplay.

## How to Use

1. Create directory `./data`
2. Add *multiple* subdirectories containing *only* MOV files. The necessity for 2+ decks stems from the fact that a model will be tested with a unique deck of cards.

Ex:
```
data
    deckName_1
        IMG_4631.MOV
        IMG_4632.MOV
        ...
        IMG_4689.MOV
    anotherDeckName
        IMG_6107.MOV
        IMG_6108.MOV
        ...
        IMG_6163.MOV
```

The first image in each subdirectory should be an ace of clubs. The next will be the two of clubs. After the king of clubs, the ace of diamonds follows. Each suit must be covered in this order: clubs, diamonds, hearts, spades. Any variation from this will make the pipeline require you to manually rename each mp4 file that each MOV file gets converted to.

3. Run `python main.py` to start the pipeline. The pipeline has several stages:

    a. The pipeline will first ask whether you have already preprocessed your dataset. If you have a directory called `processed`, you likely have processed the dataset.

    b. If you say you have *not* processed your data, the pipeline will first convert all of the MOV files into mp4 files and write these new mp4 files into a `mp4_data` directory. The deck subdirectories will also be carried over into this new directory.

    c. The pipeline will ask if your mp4 files are already named properly. This is in the case where your MOV files are already named `ace_of_clubs.MOV`, `two_of_clubs.MOV`, ..., `king_of_clubs.MOV`, `ace_of_diamonds.MOV`, and so on. If you say no, you will be asked whether your mp4 files are properly ordered as previously specified. If you say yes, your data will be automatically renamed to match the `ace_of_clubs.mp4`, etc. format. Otherwise, the pipeline will wait for you to rename your mp4 files *manually*.

    d. After obtaining properly named mp4 files, the pipeline will extract images from these videos into `raw_images`. These images will be preprocessed again and inserted into the `processed` directory, with the same deck subdirectories as in `data` and `mp4_data`.

    e. Finally, the pipeline will actually start training one or multiple models on the dataset depending on the state of the `main.py` file. Steps a-d are covered by the `pipeline.datagen` function.