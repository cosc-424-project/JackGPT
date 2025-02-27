from blackjack.blackjack import BlackJack
import matplotlib.pyplot as plt


# using basic strategy from https://www.blackjackapprenticeship.com/blackjack-strategy-charts/?bjinfo
SURRENDER_SET = set([
    "16.9", "16.10", "16.A",
    "15.10"
])
SPLIT_SET = set([
    "2.2", "2.3", "2.4", "2.5", "2.6", "2.7",
    "3.2", "3.3", "3.4", "3.5", "3.6", "3.7",
    "4.5", "4.6",
    "6.2", "6.3", "6.4", "6.5", "6.6",
    "7.2", "7.3", "7.4", "7.5", "7.6", "7.7",
    "8.2", "8.3", "8.4", "8.5", "8.6", "8.7", "8.8", "8.9", "8.10", "8.A",
    "9.2", "9.3", "9.4", "9.5", "9.6", "9.8", "9.9",
    "A.2", "A.3", "A.4", "A.5", "A.6", "A.7", "A.8", "A.9", "A.10", "A.A",
])
SOFT_DOUBLE = set([
    "2.5", "2.6",
    "3.5", "3.6",
    "4.4", "4.5", "4.6",
    "5.4", "5.5", "5.6",
    "6.3", "6.4", "6.5", "6.6",
    "7.2", "7.3", "7.4", "7.5", "7.6",
    "8.6",
])
SOFT_STAND = set([
    "7.7", "7.8",
    "8.2", "8.3", "8.4", "8.5", "8.7", "8.8", "8.9", "8.10", "8.A",
    "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8", "9.9", "9.10", "9.A",
    "10.2", "10.3", "10.4", "10.5", "10.6", "10.7", "10.8", "10.9", "10.10", "10.A"
])
HARD_DOUBLE = set([
    "9.3", "9.4", "9.5", "9.6",
    "10.2", "10.3", "10.4", "10.5", "10.6", "10.7", "10.8", "10.9",
    "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "11.9", "11.10", "11.A",
])
HARD_STAND = set([
    "12.4", "12.5", "12.6",
    "13.2", "13.3", "13.4", "13.5", "13.6",
    "14.2", "14.3", "14.4", "14.5", "14.6",
    "15.2", "15.3", "15.4", "15.5", "15.6",
    "16.2", "16.3", "16.4", "16.5", "16.6",
    "17.2", "17.3", "17.4", "17.5", "17.6", "17.7", "17.8", "17.9", "17.10", "17.A",
    "18.2", "18.3", "18.4", "18.5", "18.6", "18.7", "18.8", "18.9", "18.10", "18.A",
    "19.2", "19.3", "19.4", "19.5", "19.6", "19.7", "19.8", "19.9", "19.10", "19.A",
    "20.2", "20.3", "20.4", "20.5", "20.6", "20.7", "20.8", "20.9", "20.10", "20.A",
    "21.2", "21.3", "21.4", "21.5", "21.6", "21.7", "21.8", "21.9", "21.10", "21.A",
])


def get_set_strings(blackjack: BlackJack) -> tuple[str]:
    # get player and dealer strings
    dealer_str = "A" if type(blackjack.dealer[0].value) == tuple else str(blackjack.dealer[0].value)
    player_count_str = str(blackjack.player[blackjack.curr_hand].count())
    player_split_str = ""
    player_soft_str = ""
    if blackjack.player[blackjack.curr_hand].can_split():
        player_split_str = "A" if type(blackjack.player[blackjack.curr_hand][0].value) == tuple else str(blackjack.player[blackjack.curr_hand][0].value)
    if blackjack.player[blackjack.curr_hand].is_soft():
        player_soft_str = str(blackjack.player[blackjack.curr_hand].count() - 11)

    # combine
    count_str = player_count_str + "." + dealer_str
    split_str = ""
    soft_str = ""
    if player_split_str != "":
        split_str = player_split_str + "." + dealer_str
    if player_soft_str != "":
        soft_str = player_soft_str + "." + dealer_str
    
    return count_str, split_str, soft_str


def get_action(count_str, split_str, soft_str, can_surrender) -> str:
    '''
    first check if surrender
    then check if split
    check if hard
        check if double
        check if stand
        if neither, hit
    check if soft
        check if double
        check if stand
        if neither, hit
    '''
    if count_str in SURRENDER_SET and can_surrender:
        return "surrender"
    if split_str in SPLIT_SET:
        return "split"
    if soft_str == "":
        if count_str in HARD_DOUBLE:
            return "double"
        if count_str in HARD_STAND:
            return "stand"
        return "hit"
    else:
        if soft_str in SOFT_DOUBLE:
            return "double"
        if soft_str in SOFT_STAND:
            return "stand"
        return "hit"
    

NUM_ITER = 200000
ret_ctr = 0
win_ctr = 0
num_hands = 0

ret_avgs = []
win_avgs = []
num_hands_list = []

blackjack = BlackJack()

for i in range(NUM_ITER):
    # print()
    if not blackjack.round_running:
        blackjack.start_round()

    while blackjack.round_running:
        if not blackjack.hand_running:
            # print("hand ended:", blackjack.player[blackjack.curr_hand])
            blackjack.start_hand()

        while blackjack.hand_running:
            count_str, split_str, soft_str = get_set_strings(blackjack)
            action = get_action(count_str, split_str, soft_str, blackjack.can_surrender)

            # print(blackjack.dealer[0])
            # print(blackjack.player[blackjack.curr_hand])
            # print(action)

            if action == "hit":
                blackjack.hit()
            elif action == "stand":
                blackjack.stand()
            elif action == "surrender":
                blackjack.surrender()
            elif action == "split":
                blackjack.split()
            elif action == "double":
                blackjack.double()

    # print("hand ended:", blackjack.player[blackjack.curr_hand])
    # print(blackjack.rewards)

    num_hands += len(blackjack.rewards)
    for reward in blackjack.rewards:
        ret_ctr += reward
        if reward > 0:
            win_ctr += 1
    
    if i % 2000 == 0 and i != 0:
        ret_avgs.append(ret_ctr / num_hands)
        win_avgs.append(win_ctr / num_hands)
        num_hands_list.append(num_hands)


print(f"Win rate over {num_hands} hands: {win_ctr / num_hands : .2f}")
print(f"Average return over {num_hands} hands: {ret_ctr / num_hands : .2f}")

plt.figure(figsize=(10, 6))
plt.title("Running Win Rate")
plt.xlabel("Number of hands")
plt.ylabel("Win rate (%)")
plt.plot(num_hands_list, win_avgs)
plt.show()

plt.figure(figsize=(10, 6))
plt.title("Running Return Rate")
plt.xlabel("Number of hands")
plt.ylabel("Return rate (%)")
plt.plot(num_hands_list, ret_avgs)
plt.show()