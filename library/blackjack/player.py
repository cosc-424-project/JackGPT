from blackjack.card import Card
from blackjack.hand import Hand

class Player:
    '''Player helper class for blackjack game loop'''
    # one player can split into multiple hands
    hands: list[Hand]

    def __init__(self) -> None:
        self.hands = [Hand()]

    def __getitem__(self, key: int | tuple[int]) -> Hand:
        if type(key) != int:
            raise ValueError("Error: only 1D, integer indices are supported for Player class.")
        return self.hands[key]

    def reset(self) -> list[Card]:
        '''
        Resets player to one, empty hand
        
        Returns cards in player's hands
        '''
        cards: list[Card] = []
        for hand in self.hands:
            for card in hand:
                cards.append(card)
        self.hands = [Hand()]
        return cards
        
    def split(self, hand_idx = 0) -> None:
        '''Splits a hand, if legal—throws an error if not'''
        if not self.hands[hand_idx].can_split():
            raise ValueError(f"Cannot split hand {self.hands[hand_idx]}!")
        tmpHand = self.hands.pop(hand_idx)
        hand1 = Hand()
        hand1.push(tmpHand[0])
        hand2 = Hand()
        hand2.push(tmpHand[1])
        self.hands.insert(hand_idx, hand2)
        self.hands.insert(hand_idx, hand1)
    
    def __len__(self) -> int:
        '''Returns number of simultaneous hands'''
        return len(self.hands)
    
    def __str__(self) -> None:
        output = ""
        for i in range(len(self.hands)):
            if i != 0:
                output += "\n"
            output += str(self.hands[i])
        return output