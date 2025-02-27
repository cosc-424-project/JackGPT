from blackjack.card import Card
import random

class Deck:
    '''Deck helper class for blackjack game loop'''
    __VALUES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    __SUITS = ["D", "H", "S", "C"]
    __deck: list[Card]
    __num_decks: int
    
    def reset(self, exclude: set[Card] = {}) -> None:
        '''
        Will refill the deck with `num_decks` number of decks

        If `num_decks == 0`, then the deck will become empty
        '''
        self.__deck = []
        for i in range(self.__num_decks):
            for suit in self.__SUITS:
                for value in self.__VALUES:
                    card = Card(value + suit)
                    if card not in exclude:
                        self.__deck.append(Card(value + suit))

    def __init__(self, num_decks = 1) -> None:
        '''Use `num_decks = 0` for an empty deck'''
        self.__num_decks = num_decks
        self.reset()

    def pop(self, index: int) -> Card:
        '''Deal certain index of deck'''
        return self.__deck.pop(index)
    
    def push(self, card: Card) -> None:
        '''Add card to end of deck'''
        return self.__deck.append(card)

    def shuffle(self) -> None:
        '''Randomly shuffle internal list of cards'''
        random.shuffle(self.__deck)

    def deal(self) -> Card:
        '''Randomly deal a card from the deck'''
        idx = random.randint(0, len(self.__deck) - 1)
        return self.__deck.pop(idx)
    
    def deal_list(self, size: int) -> list[str]:
        '''Randomly deal several cards from the deck'''
        output: list[str] = []
        for i in range(size):
            output.append(self.deal())
        return output
    
    def __str__(self) -> None:
        output = "["
        for i in range(len(self.__deck)):
            output += str(self.__deck[i]) if i == 0 else f", {self.__deck[i]}"
        output += "]"
        return output
    
    def __len__(self) -> int:
        return len(self.__deck)