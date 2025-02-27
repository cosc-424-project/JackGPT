from blackjack.card import Card

class Hand:
    '''Player helper class for blackjack game loop'''
    # one player can split into multiple hands
    cards: list[Card]
    __count: int
    __has_changed: bool

    def __init__(self) -> None:
        self.__has_changed = True
        self.__count = -1
        self.cards = []

    def pop(self, index: int) -> Card:
        '''Remove certain card from a given hand'''
        self.__has_changed = True
        return self.cards.pop(index)

    def push(self, card: Card) -> None:
        '''Add certain card to a given hand'''
        self.__has_changed = True
        self.cards.append(card)

    def reset(self) -> list[Card]:
        '''Empties hand, returns cards removed'''
        cards = self.cards
        self.cards = []
        return cards

    def can_split(self) -> bool:
        '''Checks whether a given hand can split'''
        return (len(self.cards) == 2 and self.cards[0].value == self.cards[1].value)

    def count(self) -> int:
        '''
        Returns highest legal value of a given hand

        If value is over 21, the hand busted
        '''
        if not self.__has_changed:
            return self.__count

        sum = 0
        ace_present = False
        for card in self.cards:
            if type(card.value) != tuple:
                sum += card.value
            else:
                ace_present = True
                sum += 1

        if ace_present and sum + 10 <= 21:
            sum += 10

        self.__count = sum
        return sum
    
    def has_busted(self) -> bool:
        '''Returns whether a given hand has busted'''
        return self.count() > 21
    
    def has_blackjack(self) -> bool:
        '''Returns whether a given hand has a count of 21'''
        return self.count() == 21
    
    def is_soft(self) -> bool:
        '''Returns whether the highest legal value counts an ace as an 11'''
        highest_legal = self.count()
        low_sum = 0
        ace_present = False
        for card in self.cards:
            if card.value == (1, 11):
                low_sum += 1
                ace_present = True
            else:
                low_sum += card.value
        return highest_legal != low_sum and ace_present
    
    def __getitem__(self, key: int | tuple[int]) -> Card:
        if type(key) != int:
            raise ValueError("Error: only 1D, integer indices are supported for Hand class.")
        return self.cards[key]
    
    def __len__(self) -> int:
        '''Returns number of cards in given hand'''
        return len(self.cards)
    
    def __str__(self) -> None:
        output = "["
        for i in range(len(self.cards)):
            output += str(self.cards[i]) if i == 0 else f", {self.cards[i]}"
        output += f"] {self.count()}"
        return output