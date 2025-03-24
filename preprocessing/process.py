# grayscale
# 635x450 -> 630x450 (7:5 ratio) -> 63x45

import cv2
import os
from sys import argv

# check for potentially missing directories
if len(argv) != 2:
    print("usage: python process.py [folderName]")
    exit(1)
if not os.path.isdir(f"./data"):
    print("Error: data directory does not exist")
    exit(1)
if not os.path.isdir(f"./data/{argv[1]}"):
    print(f"Error: data subdirectory '{argv[1]}' does not exist")
    exit(1)

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
os.makedirs("./processed", exist_ok=True)
os.makedirs(f"./processed/{argv[1]}", exist_ok=True)

# iterate through dataset and add to data
ctr = 0
for val in vals:
    for suit in suits:
        print(f"Processing {val} of {suit}...", flush=True)
        os.makedirs(f"./processed/{argv[1]}/{val}_of_{suit}", exist_ok=True)
        for i in range(50):
            orig = cv2.imread(f"./data/{argv[1]}/{val}_of_{suit}.mp4/card_{i:02d}.png")
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray, (90, 126))
            cv2.imwrite(f"./processed/{argv[1]}/{val}_of_{suit}/card_{i:02d}.png", resized_image)
            ctr += 1