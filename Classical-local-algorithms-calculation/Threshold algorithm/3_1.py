import random

# graph 1
d = 4
neighbor1 = {
    0: (1, 2, 6, 10),
    1: (0, 2, 14, 18),
    2: (0, 1, 3, 4),
    3: (2, 50, 51, 52),
    4: (2, 53, 54, 55),
    5: (6, 47, 48, 49),
    6: (0, 5, 7, 8),
    7: (6, 8, 45, 46),
    8: (6, 7, 43, 44),
    9: (10, 40, 41, 42),
    10: (0, 9, 11, 12),
    11: (10, 12, 38, 39),
    12: (10, 11, 36, 37),
    13: (14, 32, 33, 35),
    14: (1, 13, 15, 16),
    15: (14, 16, 30, 31),
    16: (14, 15, 28, 29),
    17: (18, 25, 26, 27),
    18: (1, 17, 19, 20),
    19: (18, 20, 23, 24),
    20: (18, 19, 21, 22),
}
num_dot1 = 56

neighbor2 = {
    0: (1, 2, 3, 9),
    1: (0, 14, 17, 19),
    2: (0, 3, 4, 5),
    3: (0, 2, 6, 7),
    4: (2, 57, 58, 59),
    5: (2, 54, 55, 56),
    6: (3, 51, 52, 53),
    7: (3, 48, 49, 50),
    8: (9, 45, 46, 47),
    9: (0, 8, 10, 11),
    10: (9, 11, 43, 44),
    11: (9, 10, 41, 42),
    12: (13, 14, 39, 40),
    13: (12, 14, 37, 38),
    14: (1, 12, 13, 15),
    15: (14, 34, 35, 36),
    16: (17, 31, 32, 33),
    17: (16, 18, 1, 19),
    18: (17, 28, 29, 30),
    19: (1, 17, 20, 21),
    20: (19, 25, 26, 27),
    21: (19, 22, 23, 24),
}
num_dot2 = 60


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

                count_list.append((1 / 2 * count1 + 1 / 2 * count2) / epoch)
                para_list.append((t1, t2, t3))

    with open("p3_graph1.txt", "a") as f:
        print(max(count_list), file=f)
        idx = count_list.index(max(count_list))
        print(para_list[idx], file=f)
        print(count_list, file=f)
