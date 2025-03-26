## How to Use

1. Create directory `./data`
2. Add one or multiple subdirectories containing *only* MOV files.

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

- If the 