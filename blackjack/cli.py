from blackjack.blackjack import BlackJack


class CLI:
    blackjack: BlackJack
    debug: bool


    def __init__(self, debug = False):
        self.blackjack = BlackJack()
        self.debug = debug


    def handle_cmd(self, args: list[str]) -> bool:
        '''Returns whether command was requested in valid context'''
        if args[0] == "insurance":
            if not self.blackjack.can_insurance:
                print("You can only play insurance at first glance when dealer shows an ace.")
                return False
            if len(args) == 2:
                try:
                    self.blackjack.insurance(float(args[1]))
                except:
                    raise ValueError("Failed to parse insurance bet as float.")
            elif len(args) == 1:
                self.blackjack.insurance()
            else:
                raise ValueError("usage: insurance [bet]")
        elif args[0] == "hit":
            self.blackjack.hit()
        elif args[0] == "stand":
            self.blackjack.stand()
        elif args[0] == "surrender":
            if self.blackjack.can_surrender:
                self.blackjack.surrender()
            else:
                print("You cannot surrender after hitting or splitting.")
                return False
        elif args[0] == "split":
            self.blackjack.split()
        elif args[0] == "double":
            self.blackjack.double()
        return True


    def loop(self):
        '''CLI loop for playing moves and displaying game state'''
        win_ctr = 0
        ret_ctr = 0
        num_hands = 0

        line = ""
        valid_cmd = True
        while line != "quit" and line != "exit":
            have_started_new_round = False
            if not self.blackjack.round_running:
                print("Final Player:")
                print(self.blackjack.player)
                print("Final Dealer:", self.blackjack.dealer)
                print("Rewards:", self.blackjack.rewards)
                print()

                # add rewards to ctr
                num_hands += len(self.blackjack.rewards)
                for reward in self.blackjack.rewards:
                    ret_ctr += reward
                    if reward > 0:
                        win_ctr += 1
                
                self.blackjack.start_round()
                have_started_new_round = True
                
            if not self.blackjack.hand_running:
                if not have_started_new_round:
                    print(f"Final Hand {self.blackjack.curr_hand}:", self.blackjack.player[self.blackjack.curr_hand])
                self.blackjack.start_hand()

            if line in ["", "quit", "exit", "insurance", "hit", "stand", "surrender", "split", "double"] and valid_cmd:
                print("Dealer:", self.blackjack.dealer[0], self.blackjack.dealer[0].value)
                print(f"Hand {self.blackjack.curr_hand}:", self.blackjack.player[self.blackjack.curr_hand])
                if self.debug:
                    print("discard size: ", len(self.blackjack.discard))
                    print("deck size: ", len(self.blackjack.deck))
                    print("active size: ", len(self.blackjack.active_cards))
                    print("total size: ", len(self.blackjack.discard) + len(self.blackjack.deck) + len(self.blackjack.active_cards))
            line = input("> ")

            valid_cmd = self.handle_cmd([arg for arg in line.split(" ") if arg])
    
        if (num_hands > 0):
            print("\nFINAL RESULTS:")
            print(f"Win rate over {num_hands} hands: {win_ctr / num_hands : .2f}")
            print(f"Return rate over {num_hands} hands: {ret_ctr / num_hands : .2f}")