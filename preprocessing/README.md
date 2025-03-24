# Preprocessing

A preprocessing program which converts every card image into a smaller, grayscale image of 126x90.

## How to Use

Add a directory called `data`, and add another directory within here which includes all of the images.

Ex:
```
preprocessing
    data
        deck_01
            ace_of_clubs.mp4
            ace_of_diamonds.mp4
            ...
            king_of_spades.mp4
```

Running `python process.py deck_01` wll create a new directory
called processed which has the following file architecture:

```
preprocessing
    processed
        deck_01
            ace_of_clubs
            ace_of_diamonds
            ...
            king_of_spades
```
Note that, in both of these scenarios, `deck_01` can be any folder name!