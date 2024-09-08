import random

# graph 2
neighbor = {
    0: (1, 2, 3, 4),
    1: (0, 5, 6, 7)
}

d = 4
num_dot = 8

epoch = 1000000


def flip(k):
    if k == 0:
        return 1
    else:
        return 0


def deal(c_list,tao):
    fliplist = []
    for key,value in neighbor.items():
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
        count = 0

        for _ in range(epoch):
            coin_list = [random.randint(0, 1) for _ in range(num_dot)]
            coin_list = deal(coin_list, t1)
            if coin_list[0] != coin_list[1]:
                count += 1

        count_list.append(count / epoch)
        para_list.append(t1)

    with open("p1_graph2.txt", "a") as f:
        print(max(count_list), file=f)
        idx = count_list.index(max(count_list))
        print(para_list[idx], file=f)
        print(count_list, file=f)
