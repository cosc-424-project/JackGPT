from sys import stderr

# Each rank of card
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

# Each suit in a deck
CARD_SUITS = [
    "clubs",
    "diamonds",
    "hearts",
    "spades",
]

def confirm(prompt: str) -> bool:
    '''
    Will submit a prompt and require a yes/no response.
    '''
    res = ""
    while res != "y" and res != "yes" and res != "n" and res != "no":
        print(prompt, end="")
        res = input().lower()
    return res == "y" or res == "yes"

def error(message: str) -> None:
    '''
    Print a error message to stderr and return with an exit code of `1`.
    '''
    print(message, file=stderr)
    exit(1)