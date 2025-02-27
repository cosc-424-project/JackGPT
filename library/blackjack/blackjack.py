from blackjack.deck import Deck
from blackjack.player import Player
from blackjack.card import Card
from blackjack.hand import Hand


class BlackJack:
    '''Simulates games of 1-on-1 BlackJack with HiLo counting, history, infinite splits, surrender, insurance, and double down'''
    

    # user preferences

    win_ratio: float
    hit_soft_17: bool
    charlie: bool

    # hand independent variables

    round_running: bool
    hand_running: bool
    player: Player
    dealer: Hand
    deck: Deck
    discard: Deck
    active_cards: set[Card]
    hilo: float

    # hand dependent variables

    curr_hand: int
    insurances: list[float]
    surr_double: list[str]
    rewards: list[float]
    can_surrender: bool
    can_insurance: bool


    # constructor

    def __init__(self, win_ratio = 1.5, hit_soft_17 = True, charlie = True) -> None:
        '''Automatically starts first round and hand'''
        self.win_ratio = win_ratio
        self.hit_soft_17 = hit_soft_17

        self.charlie = charlie
        self.hand_running = False
        self.round_running = False
        self.dealer = Hand()
        self.player = Player()
        self.deck = Deck()
        self.discard = Deck(0)
        self.active_cards = set()
        self.hilo = 0

        self.start_round()
        self.start_hand()


    # card management functions

    def __get_card(self) -> Card:
        '''Gets card from deck, reshuffles if necessary, and updates HiLo'''
        card = self.deck.deal()
        self.active_cards.add(card)
        if len(self.deck) == 0:
            self.deck.reset(self.active_cards)
            self.discard.reset()
            self.hilo = 0
        if type(card.value) == tuple or card.value == 10:
            self.hilo -= 1
        elif card.value <= 6:
            self.hilo += 1
        return card

    def __discard(self) -> None:
        '''Clears player's and dealer's cards and adds them to discard pile'''
        self.active_cards.clear()
        for card in self.player.reset():
            self.discard.push(card)
        for card in self.dealer.reset():
            self.discard.push(card)

    def reset(self) -> None:
        '''Reshuffle the deck with discard pile outside of any round'''
        if self.round_running or self.hand_running:
            raise RuntimeError("Can only manually reset deck outside of any round.")
        self.deck.reset(self.active_cards)
        self.discard.reset()
        self.hilo = 0


    # reward functions

    def __play_dealer(self) -> None:
        # dealer's second card revealed, update HiLo
        if type(self.dealer[1].value) == tuple or self.dealer[1].value == 10:
            self.hilo -= 1
        elif self.dealer[1].value <= 6:
            self.hilo += 1

        while self.dealer.count() < 17:
            self.dealer.push(self.__get_card())
        if self.dealer.count() == 17 and self.dealer.is_soft() and self.hit_soft_17:
            self.dealer.push(self.__get_card())

    def __update_rewards(self) -> None:
        '''Fills rewards variable with return for each hand'''
        '''
        End States:
        user surrendered = half lose
        user 5-charlie
        user busted = lose
        user doubled and busted = double lose

        dealer busts
            user stood = win
            user doubled and didn't bust = double win
        dealer didn't bust
            user ties = tie
            user stood and won = win
            user doubled and won = double win
            user lost = lose
            user doubled and lost = double loss

        user had insurance and dealer had blackjack
        '''
        self.__play_dealer()

        # iterate through different end cases
        for i in range(len(self.player)):
            # check if surrendered
            if self.surr_double[i] == "surrendered":
                self.rewards.append(-.5)

            elif len(self.player[i]) == 5 and self.player[i].count() < 21 and self.charlie:
                self.rewards.append(1.0)

            # check if busted
            elif self.player[i].has_busted():
                if self.surr_double[i] == "doubled":
                    self.rewards.append(-2)
                else:
                    self.rewards.append(-1)
            
            # if dealer busted AND user not busted
            elif self.dealer.has_busted():
                if self.surr_double[i] == "doubled":
                    self.rewards.append(2 * self.win_ratio)
                else:
                    self.rewards.append(self.win_ratio)
            
            # if neither dealer nor user busted
            else:
                if self.player[i].count() == self.dealer.count():
                    self.rewards.append(0.)
                elif self.player[i].count() > self.dealer.count():
                    if self.surr_double[i] == "doubled":
                        self.rewards.append(2 * self.win_ratio)
                    else:
                        self.rewards.append(self.win_ratio)
                else:
                    if self.surr_double[i] == "doubled":
                        self.rewards.append(-2.)
                    else:
                        self.rewards.append(-1.)
            
            # if there's insurance, add that if needed
            if self.dealer.has_blackjack() and len(self.dealer) == 2:
                self.rewards[-1] += 2 * self.insurances[i]
            else:
                self.rewards[-1] -= self.insurances[i]
        

    # start and end functions

    def start_round(self) -> None:
        '''Empties player's hands and gives two cards to dealer'''
        if self.round_running:
            raise RuntimeError("Blackjack round already running.")

        self.curr_hand = -1
        self.__discard()
        self.player[0].push(self.__get_card())
        self.player[0].push(self.__get_card())
        self.dealer.push(self.__get_card())
        self.dealer.push(self.__get_card())

        # correct HiLo to ignore second dealer card
        if type(self.dealer[1].value) == tuple or self.dealer[1].value == 10:
            self.hilo += 1
        elif self.dealer[1].value <= 6:
            self.hilo -= 1

        # check dealer card for insurance
        self.can_insurance = False
        if type(self.dealer[0].value) == tuple:
            self.can_insurance = True

        self.round_running = True
        self.rewards = []
        self.insurances = []
        self.surr_double = []

    def start_hand(self) -> None:
        '''Begin accepting play functions'''
        if self.hand_running:
            raise RuntimeError("Blackjack hand already running.")
        
        self.curr_hand += 1
        self.insurances.append(0.)
        self.surr_double.append("")
        self.hand_running = True
        self.can_surrender = True

    def __end_round(self) -> None:
        self.round_running = False
        self.__update_rewards()

    def __end_hand(self) -> None:
        '''Sets hand_running to False and ends game if no hands left'''
        self.hand_running = False
        if self.curr_hand + 1 == len(self.player):
            self.__end_round()
    

    # play functions

    def insurance(self, bet = .5) -> None:
        '''Changes insurance bet ()'''
        if not self.can_insurance:
            raise RuntimeError("Can only play insurance at first glance when dealer shows an ace.")
        if bet < 0. or bet > .5:
            raise ValueError("Insurance bet cannot be more than .5 or less than 0.")
        self.insurances[self.curr_hand] = bet
    
    def hit(self) -> None:
        '''Deals card to player, *can* terminate the hand'''
        if not (self.hand_running and self.round_running):
            raise RuntimeError("Must have round and hand running to play.")
        self.can_surrender = False
        self.can_insurance = False
        card = self.__get_card()
        self.player[self.curr_hand].push(card)
        is_charlie = len(self.player[self.curr_hand]) == 5 and self.player[self.curr_hand].count() < 21 and self.charlie
        if self.player[self.curr_hand].has_busted():
            self.__end_hand()
        elif is_charlie:
            self.__end_hand()
    
    def stand(self) -> None:
        '''Terminates the hand'''
        if not (self.hand_running and self.round_running):
            raise RuntimeError("Must have round and hand running to play.")
        self.can_surrender = False
        self.can_insurance = False
        self.__end_hand()
    
    def surrender(self) -> None:
        '''Terminates the hand, setting half loss'''
        if not self.can_surrender:
            raise RuntimeError("Cannot surrender after hitting or splitting.")
        if not (self.hand_running and self.round_running):
            raise RuntimeError("Must have round and hand running to play.")
        self.surr_double[self.curr_hand] = "surrendered"
        self.__end_hand()
    
    def split(self) -> None:
        '''Split current hand into two, automatically starting one of the children'''
        if not (self.hand_running and self.round_running):
            raise RuntimeError("Must have round and hand running to play.")
        self.can_surrender = False
        self.can_insurance = False
        self.player.split(self.curr_hand)
        self.player[self.curr_hand].push(self.__get_card())
        self.player[self.curr_hand + 1].push(self.__get_card())
    
    def double(self) -> None:
        '''Deals card to player, terminates the hand, and sets double down'''
        if not (self.hand_running and self.round_running):
            raise RuntimeError("Must have round and hand running to play.")
        self.can_surrender = False
        self.can_insurance = False
        card = self.__get_card()
        self.player[self.curr_hand].push(card)
        self.surr_double[self.curr_hand] = "doubled"
        self.__end_hand()