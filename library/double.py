from blackjack.blackjack import BlackJack
from sys import argv


debug = False
if len(argv) == 2 and argv[1] == "debug":
    debug = True


NUM_ITER = 200000
ret_ctr = 0.0
win_ctr = 0.0
blackjack = BlackJack()
for i in range(NUM_ITER):
    if not blackjack.round_running:
        blackjack.start_round()
        blackjack.start_hand()

    # print state
    if debug:
        print("Dealer:", blackjack.dealer)
        print("Player:", blackjack.player)

    blackjack.double()

    if debug:
        print("Final dealer:", blackjack.dealer)
        print("Reward:", blackjack.rewards[0])
        print()

    ret_ctr += blackjack.rewards[0]
    if blackjack.rewards[0] > 0:
        win_ctr += 1

print(f"Win rate when only standing for {NUM_ITER} games: {win_ctr / NUM_ITER : .2f}")
print(f"Average return when only standing for {NUM_ITER} games: {ret_ctr / NUM_ITER : .2f}")