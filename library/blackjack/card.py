from typing import Self

class Card:
    '''Card helper class for blackjack game loop'''
    name: str
    value: int | tuple[int]
    suit: str

    def __init__(self, name: str):
        val10s = ["10", "J", "K", "Q"]
        self.name = name
        val_str = self.name[:-1]
        if val_str in val10s:
            self.value = 10
        elif val_str == "A":
            self.value = (1, 11)
        else:
            self.value = int(val_str)
        self.suit = self.name[-1]
    
    def __str__(self):
        return self.name
    
    def __eq__(self, other: Self):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)