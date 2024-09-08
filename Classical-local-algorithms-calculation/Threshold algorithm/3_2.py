import random
from numba import jit

# graph 2
neighbor = {
    0: (1, 2, 3, 4),
    1: (0, 5, 6, 7),
    2: (0, 8, 9, 10),
    3: (0, 8, 11, 12),
    4: (0, 7, 13, 14),
    5: (1, 15, 16, 17),
    6: (1, 15, 18, 19),
    7: (1, 4, 20, 21),
    8: (2, 3, 44, 45),
    9: (2, 46, 47, 48),
    10: (2, 48, 49, 50),
    11: (3, 41, 42, 43),
    12: (3, 39, 40, 41),
    13: (4, 51, 52, 53),
    14: (4, 53, 54, 55),
    15: (5, 6, 32, 33),
    16: (5, 27, 28, 29),
    17: (5, 29, 30, 31),
    18: (6, 36, 37, 38),
    19: (6, 34, 35, 36),
    20: (7, 22, 23, 24),
    21: (7, 24, 25, 26),
}

d = 4
num_dot = 56

epoch = 100000


def flip(k):
    if k == 0:
        return 1
    else:
        return 0

# @jit
def deal(c_list, tao):
    fliplist = []
    for key, value in neighbor.items():
        nei_num = 0
        for i in value:
            nei_num += coin_list[i]
        if coin_list[key] == 0:
            nei_num = d - nei_num

        if nei_num > tao:
            fliplist.append(key)

    for coin in fliplist:
        c_list[coin] = flip(c_list[coin])

    return c_list


for i in range(100):
    count_list = []
    para_list = []

    for t1 in range(0, d + 2):
        for t2 in range(0, d + 2):
            for t3 in range(0, d + 2):
                count = 0

                for _ in range(epoch):
                    coin_list = [random.randint(0, 1) for _ in range(num_dot)]
                    coin_list = deal(coin_list, t1)
                    coin_list = deal(coin_list, t2)
                    coin_list = deal(coin_list, t3)
                    if coin_list[0] != coin_list[1]:
                        count += 1

                count_list.append(count / epoch)
                para_list.append((t1, t2, t3))

    with open("p3_graph2.txt", "a") as f:
        print(max(count_list), file=f)
        idx = count_list.index(max(count_list))
        print(para_list[idx], file=f)
        print(count_list, file=f)
