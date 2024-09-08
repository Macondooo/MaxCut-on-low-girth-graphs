import numpy as np
import pandas as pd
from threading import Thread
from multiprocessing import Process

n_samples = 10000000


def expected_cut_fraction_1local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        # non-cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, C, U, V = np.random.normal(0, 1, 5)
            if (
                U + para[0] * (2**0.5 * B + A + V) > 0
                and V + para[0] * (2**0.5 * C + A + U) < 0
            ):
                correct_nums += 1

        g11_1 = correct_nums / n_samples * 2

        # cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, U, V = np.random.normal(0, 1, 4)
            if (
                U + para[0] * (3**0.5 * A + V) > 0
                and V + para[0] * (3**0.5 * B + U) < 0
            ):
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
            A, B, C, D, E, F, U, V = np.random.normal(0, 1, 8)
            if (
                U
                + para[0] * (A + V + 2**0.5 * B)
                + para[1] * (2**0.5 * (D + C) + 6**0.5 * E)
                > 0
                and V
                + para[0] * (A + U + 2**0.5 * C)
                + para[1] * (2**0.5 * (D + B) + 6**0.5 * F)
                < 0
            ):
                correct_nums += 1

        g11_2 = correct_nums / n_samples * 2

        # cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, C, D, U, V = np.random.normal(0, 1, 6)
            if (
                U + para[0] * (3**0.5 * A + V) + para[1] * (7**0.5 * B + 3**0.5 * C) > 0
                and V + para[0] * (U + 3**0.5 * C) + para[1] * (7**0.5 * D + 3**0.5 * A)
                < 0
            ):
                correct_nums += 1

        g12_2 = correct_nums / n_samples * 2

        result_list.append((g11_2 + g12_2) / 2)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


def expected_cut_fraction_3local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        # non-cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, C, D, E, F, G, H, I, U, V = np.random.normal(0, 1, 11)
            if (
                U
                + para[0] * (A + V + 2**0.5 * B)
                + para[1] * (2**0.5 * (D + C) + 6**0.5 * E)
                + para[2] * (14**0.5 * G + 6**0.5 * I + 6**0.5 * F)
                > 0
                and V
                + para[0] * (A + U + 2**0.5 * C)
                + para[1] * (2**0.5 * (D + B) + 6**0.5 * F)
                + para[2] * (14**0.5 * H + 6**0.5 * I + 6**0.5 * E)
                < 0
            ):
                correct_nums += 1

        g11_3 = correct_nums / n_samples * 2

        # cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, C, D, E, F, U, V = np.random.normal(0, 1, 8)
            if (
                U
                + para[0] * (3**0.5 * A + V)
                + para[1] * (7**0.5 * B + 3**0.5 * C)
                + para[2] * (19**0.5 * E + 7**0.5 * D)
                > 0
                and V
                + para[0] * (U + 3**0.5 * C)
                + para[1] * (7**0.5 * D + 3**0.5 * A)
                + para[2] * (19**0.5 * F + 7**0.5 * B)
                < 0
            ):
                correct_nums += 1

        g12_3 = correct_nums / n_samples * 2

        result_list.append((g11_3 + g12_3) / 2)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


para1 = [-(3 ** (-0.5)), 3 ** (-1), -(3 ** (-1.5))]
para2 = [-(2 ** (-1)), 10 ** (-0.5), -(26 ** (-0.5))]
para3_1 = [-(2 ** (-0.5)), 6 ** (-0.5), -(14 ** (-0.5))]
para3_2 = [-(3 ** (-0.5)), 7 ** (-0.5), -(19 ** (-0.5))]
para4_1 = [-(3 ** (-0.5)), 8 ** (-0.5), -(20 ** (-0.5))]
para4_2 = [-(3 ** (-0.5)), 7 ** (-0.5), -(19 ** (-0.5))]

paras = [para1, para2, para3_1, para3_2, para4_1, para4_2]
para_names = ["1", "2", "3_1", "3_2", "4_1", "4_2"]
funcs = [
    expected_cut_fraction_1local,
    expected_cut_fraction_2local,
    expected_cut_fraction_3local,
]
# start thread
for i, func in enumerate(funcs):
    for name, para in zip(para_names, paras):
        path = "graph1_output/10m/" + str(i + 1) + "local_para" + name
        thread = Process(target=func, args=(path, para, 50))
        thread.start()

# Thread(target=expected_cut_fraction_3local,args=('out.txt', paras[2], 50))
