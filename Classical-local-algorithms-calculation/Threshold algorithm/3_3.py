import random

# graph 3
d = 3
neighbor1 = {
    0: (1, 2, 3),
    1: (0, 2, 4),
    2: (0, 1, 9),
    3: (0, 5, 6),
    4: (1, 7, 8),
    5: (3, 6, 15),
    6: (3, 5, 14),
    7: (4, 8, 13),
    8: (4, 7, 12),
    9: (2, 10, 11),
}
num_dot1 = 16

neighbor2 = {
    0: (1, 2, 3),
    1: (0, 5, 4),
    2: (0, 3, 6),
    3: (0, 2, 7),
    4: (1, 5, 8),
    5: (1, 4, 9),
    6: (2, 16, 17),
    7: (3, 14, 15),
    8: (4, 12, 13),
    9: (5, 10, 11),
}
num_dot2 = 18


epoch = 100000


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
            for t3 in range(0, d + 2):
                count1 = 0
                for _ in range(epoch):
                    coin_list1 = [random.randint(0, 1) for _ in range(num_dot1)]
                    coin_list1 = deal(coin_list1, t1, neighbor1)
                    coin_list1 = deal(coin_list1, t2, neighbor1)
                    coin_list1 = deal(coin_list1, t3, neighbor1)
                    if coin_list1[0] != coin_list1[1]:
                        count1 += 1

                count2 = 0
                for _ in range(epoch):
                    coin_list2 = [random.randint(0, 1) for _ in range(num_dot2)]
                    coin_list2 = deal(coin_list2, t1, neighbor2)
                    coin_list2 = deal(coin_list2, t2, neighbor2)
                    coin_list2 = deal(coin_list2, t3, neighbor2)
                    if coin_list2[0] != coin_list2[1]:
                        count2 += 1

                count_list.append((2 / 3 * count1 + 1 / 3 * count2) / epoch)
                para_list.append((t1, t2, t3))

    with open("p3_graph3.txt", "a") as f:
        print(max(count_list), file=f)
        idx = count_list.index(max(count_list))
        print(para_list[idx], file=f)
        print(count_list, file=f)
