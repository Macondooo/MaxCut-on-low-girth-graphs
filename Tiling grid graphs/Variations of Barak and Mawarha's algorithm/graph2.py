import numpy as np
import pandas as pd
from threading import Thread
from multiprocessing import Process

n_samples = 10000000


def expected_cut_fraction_1local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        # edge 1
        correct_nums = 0
        for i in range(n_samples):
            A, B, D, H, G, E, U, V = np.random.normal(0, 1, 8)
            if U + para[0] * (B + A + D + V) > 0 and V + para[0] * (H + G + E + U) < 0:
                correct_nums += 1

        g11_1 = correct_nums / n_samples * 2

        # edge 2
        correct_nums = 0
        for i in range(n_samples):
            A, B, I, C, J, U, V = np.random.normal(0, 1, 7)
            if U + para[0] * (A + V + I + B) > 0 and V + para[0] * (A + U + C + J) < 0:
                correct_nums += 1

        g12_1 = correct_nums / n_samples * 2

        result_list.append((g11_1 + g12_1) / 2)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


def expected_cut_fraction_2local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        # non-cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, C, D, E, F, G, H, I, U, V, N, K, L, M, J = np.random.normal(0, 1, 16)
            if (
                U
                + para[0] * (B + A + D + V)
                + para[1] * (K + L + C + 2**0.5 * M + E + G + H)
                > 0
                and V
                + para[0] * (H + G + E + U)
                + para[1] * (J + I + F + 2**0.5 * N + B + D + A)
                < 0
            ):
                correct_nums += 1

        g11_2 = correct_nums / n_samples * 2

        # cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, C, D, E, F, G, H, I, J, K, L, U, V = np.random.normal(0, 1, 14)
            if (
                U
                + para[0] * (A + V + I + B)
                + para[1] * (H + G + J + C + D + E + 2**0.5 * K)
                > 0
                and V
                + para[0] * (A + U + C + J)
                + para[1] * (2**0.5 * L + H + G + I + B + E + F)
                < 0
            ):
                correct_nums += 1

        g12_2 = correct_nums / n_samples * 2

        result_list.append((g11_2 + g12_2) / 2)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


para1 = [-(3 ** (-0.5)), 3 ** (-1)]
para2 = [-(4 ** (-0.5)), 8 ** (-0.5)]
para3_1 = [-(3 ** (-0.5)), 5 ** (-0.5)]
para3_2 = [-(2 ** (-0.5)), 4 ** (-0.5)]
para4_1 = [-(3 ** (-0.5)), 5 ** (-0.5)]
para4_2 = [-(3 ** (-0.5)), 5 ** (-0.5)]

paras = [para1, para2, para3_1, para3_2, para4_1, para4_2]
para_names = ["1", "2", "3_1", "3_2", "4_1", "4_2"]
funcs = [expected_cut_fraction_1local, expected_cut_fraction_2local]
# start thread
for i, func in enumerate(funcs):
    for name, para in zip(para_names, paras):
        path = "graph2_output/" + str(i + 1) + "local_para" + name
        thread = Process(target=func, args=(path, para, 50))
        thread.start()

# Thread(target=expected_cut_fraction_3local,args=('out.txt', paras[2], 50))
