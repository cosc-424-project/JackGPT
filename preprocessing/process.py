# grayscale
# 635x450 -> 630x450 (7:5 ratio) -> 63x45

import cv2
import os

vals = [
    "ace",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "jack",
    "queen",
    "king",
]

suits = [
    "clubs",
    "diamonds",
    "hearts",
    "spades",
]

# create new data directory
os.makedirs("processed", exist_ok=True)

# iterate through dataset and add to data
ctr = 0
for val in vals:
    for suit in suits:
        print(f"Processing {val} of {suit}...", flush=True)
        os.makedirs(f"processed/{val}_of_{suit}", exist_ok=True)
        for i in range(50):
            orig = cv2.imread(f"data/{val}_of_{suit}.mp4/card_{i:02d}.png")
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray, (90, 126))
            cv2.imwrite(f"processed/{val}_of_{suit}/card_{i:02d}.png", resized_image)
            ctr += 1