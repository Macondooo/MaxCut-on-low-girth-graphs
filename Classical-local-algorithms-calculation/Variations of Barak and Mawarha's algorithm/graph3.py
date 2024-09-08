import numpy as np
import pandas as pd
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
            if U + para[0] * (A + V + C) > 0 and V + para[0] * (B + U + C) < 0:
                correct_nums += 1

        g31_1 = correct_nums / n_samples * 2

        # cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, U, V = np.random.normal(0, 1, 4)
            if (
                U + para[0] * (2**0.5 * A + V) > 0
                and V + para[0] * (2**0.5 * B + U) < 0
            ):
                correct_nums += 1

        g32_1 = correct_nums / n_samples * 2

        result_list.append(g31_1 * 2 / 3 + g32_1 * 1 / 3)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


def expected_cut_fraction_2local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        # non-cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, C, U, V, D, E, F = np.random.normal(0, 1, 8)
            if (
                U + para[0] * (A + V + C) + para[1] * (2**0.5 * D + F + B) > 0
                and V + para[0] * (B + U + C) + para[1] * (2**0.5 * E + F + A) < 0
            ):
                correct_nums += 1

        g31_2 = correct_nums / n_samples * 2

        # cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, U, V, C, D = np.random.normal(0, 1, 6)
            if (
                U + para[0] * (2**0.5 * A + V) + para[1] * (2**0.5 * C + 2**0.5 * B) > 0
                and V + para[0] * (2**0.5 * B + U) + para[1] * (2**0.5 * D + 2**0.5 * A)
                < 0
            ):
                correct_nums += 1

        g32_2 = correct_nums / n_samples * 2

        result_list.append(g31_2 * 2 / 3 + g32_2 * 1 / 3)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


def expected_cut_fraction_3local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        # non-cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, C, U, V, D, E, F, G, H, I = np.random.normal(0, 1, 11)
            if (
                U
                + para[0] * (A + V + C)
                + para[1] * (2**0.5 * D + F + B)
                + para[2] * (2**0.5 * G + 2**0.5 * I + 2**0.5 * E)
                > 0
                and V
                + para[0] * (B + U + C)
                + para[1] * (2**0.5 * E + F + A)
                + para[2] * (2**0.5 * H + 2**0.5 * I + 2**0.5 * D)
                < 0
            ):
                correct_nums += 1

        g31_3 = correct_nums / n_samples * 2

        # cut edge
        correct_nums = 0
        for i in range(n_samples):
            A, B, U, V, C, D, E, F = np.random.normal(0, 1, 8)
            if (
                U
                + para[0] * (2**0.5 * A + V)
                + para[1] * (2**0.5 * C + 2**0.5 * B)
                + para[2] * (4**0.5 * E + 2**0.5 * D)
                > 0
                and V
                + para[0] * (2**0.5 * B + U)
                + para[1] * (2**0.5 * D + 2**0.5 * A)
                + para[2] * (4**0.5 * F + 2**0.5 * C)
                < 0
            ):
                correct_nums += 1

        g32_3 = correct_nums / n_samples * 2

        result_list.append(g31_3 * 2 / 3 + g32_3 * 1 / 3)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


para1 = [-(2 ** (-0.5)), 2 ** (-1), -(2 ** (-1.5))]
para2 = [-(3 ** (-0.5)), 4 ** (-0.5), -(6 ** (-0.5))]
para3_1 = [-(1 ** (-0.5)), 2 ** (-0.5), -(2 ** (-0.5))]
para3_2 = [-(2 ** (-0.5)), 2 ** (-0.5), -(4 ** (-0.5))]
para4_1 = [-(2 ** (-0.5)), 3 ** (-0.5), -(4 ** (-0.5))]
para4_2 = [-(2 ** (-0.5)), 2 ** (-0.5), -(4 ** (-0.5))]

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
        path = "graph3_output/10m/" + str(i + 1) + "local_para" + name
        thread = Process(target=func, args=(path, para, 50))
        thread.start()

# Thread(target=expected_cut_fraction_3local,args=('out.txt', paras[2], 50))
