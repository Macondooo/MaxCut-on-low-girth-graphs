import random
from numba import jit

# graph 1
d = 4
neighbor1 = {0:(1,2,3,6),
             1:(0,4,5,6)
            }
num_dot1 = 7

neighbor2 = {0:(1,2,3,4),
             1:(0,5,6,7)
            }
num_dot2 = 8


epoch = 1000000

@jit
def flip(k):
    if k == 0:
        return 1
    else:
        return 0

@jit
def deal(c_list,tao,neighbor):
    fliplist = []
    for key,value in neighbor.items():
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
    count_list=[]
    for t1 in range(0,d+2):

        count1 = 0
        for _ in range(epoch):
            coin_list1 = [random.randint(0,1) for _ in range(num_dot1)]
            coin_list1 = deal(coin_list1,t1,neighbor1)
            if coin_list1[0] != coin_list1[1]:
                count1 += 1
                
        count2 = 0
        for _ in range(epoch):
            coin_list2 = [random.randint(0,1) for _ in range(num_dot2)]
            coin_list2 = deal(coin_list2,t1,neighbor2)
            if coin_list2[0] != coin_list2[1]:
                count2 += 1

        count_list.append((1/2*count1+1/2*count2)/epoch)

    with open("p1_graph1.txt", "a") as f:
        print(max(count_list),file=f)
        idx = count_list.index(max(count_list)) 
        print(idx,file=f)
        print(count_list,file=f)