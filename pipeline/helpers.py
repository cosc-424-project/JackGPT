from sys import stderr

CARD_VALS = [
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

CARD_SUITS = [
    "clubs",
    "diamonds",
    "hearts",
    "spades",
]

def confirm(prompt: str) -> bool:
    res = ""
    while res != "y" and res != "yes" and res != "n" and res != "no":
        print(prompt, end="")
        res = input().lower()
    return res == "y" or res == "yes"

def error(message: str) -> None:
    print(message, file=stderr)
    exit(1)