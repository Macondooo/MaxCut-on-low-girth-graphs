import random
from numba import jit

# graph 1
d = 3
neighbor1 = {
    0: (1, 2, 3),
    1: (0, 4, 5),
    2: (0, 7, 6),
    3: (0, 8, 9),
    4: (1, 9, 10),
    5: (1, 11, 12),
}
num_dot1 = 13

neighbor2 = {
    0: (1, 2, 3),
    1: (0, 4, 5),
    2: (0, 7, 6),
    3: (0, 8, 9),
    4: (1, 10, 11),
    5: (1, 12, 13),
}
num_dot2 = 14


epoch = 1000000


def flip(k):
    if k == 0:
        return 1
    else:
        return 0


def deal(c_list, tao, neighbor):
    fliplist = []
    for key, value in neighbor.items():
        nei_num = 0
        for i in value:
            nei_num += c_list[i]
        if c_list[key] == 0:
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

            count1 = 0
            for _ in range(epoch):
                coin_list1 = [random.randint(0, 1) for _ in range(num_dot1)]
                coin_list1 = deal(coin_list1, t1, neighbor1)
                if coin_list1[0] != coin_list1[1]:
                    count1 += 1

            count2 = 0
            for _ in range(epoch):
                coin_list2 = [random.randint(0, 1) for _ in range(num_dot2)]
                coin_list2 = deal(coin_list2, t1, neighbor2)
                if coin_list2[0] != coin_list2[1]:
                    count2 += 1

            count_list.append((2 / 3 * count1 + 1 / 3 * count2) / epoch)
            para_list.append((t1, t2))

    with open("p2_graph1.txt", "a") as f:
        print(max(count_list), file=f)
        idx = count_list.index(max(count_list))
        print(para_list[idx], file=f)
        print(count_list, file=f)
