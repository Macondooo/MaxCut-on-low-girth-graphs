import numpy as np
import pandas as pd
from multiprocessing import Process

n_samples = 10000000


def expected_cut_fraction_1local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        correct_nums = 0

        for i in range(n_samples):
            A, B, E, F, U, V = np.random.normal(0, 1, 6)
            if (
                U + para[0] * (2**0.5 * A + V + E) > 0
                and V + para[0] * (2**0.5 * B + U + F) < 0
            ):
                correct_nums += 1

        result_list.append(correct_nums / n_samples * 2)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


def expected_cut_fraction_2local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        correct_nums = 0

        for i in range(n_samples):
            A, B, E, F, C, D, G, H, U, V = np.random.normal(0, 1, 10)
            if (
                U
                + para[0] * (2**0.5 * A + V + E)
                + para[1] * (5**0.5 * C + 2**0.5 * G + F + 2**0.5 * B)
                > 0
                and V
                + para[0] * (2**0.5 * B + U + F)
                + para[1] * (5**0.5 * D + 2**0.5 * H + E + 2**0.5 * A)
                < 0
            ):
                correct_nums += 1

        result_list.append(correct_nums / n_samples * 2)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


def expected_cut_fraction_3local(path, para, repeated_times):
    result_list = []
    for i in range(repeated_times):
        print(path + ": repeated times " + str(i))
        correct_nums = 0

        for i in range(n_samples):
            A, B, E, F, C, D, G, H, U, V, I, J, K, L = np.random.normal(0, 1, 14)
            if (
                U
                + para[0] * (2**0.5 * A + V + E)
                + para[1] * (5**0.5 * C + 2**0.5 * G + F + 2**0.5 * B)
                + para[2] * (12**0.5 * K + 5**0.5 * I + 2**0.5 * H + 5**0.5 * D)
                > 0
                and V
                + para[0] * (2**0.5 * B + U + F)
                + para[1] * (5**0.5 * D + 2**0.5 * H + E + 2**0.5 * A)
                + para[2] * (12**0.5 * L + 5**0.5 * J + 2**0.5 * G + 5**0.5 * C)
                < 0
            ):
                correct_nums += 1

        result_list.append(correct_nums / n_samples * 2)

    data = pd.DataFrame(data=result_list, index=None, columns=["expected cut fraction"])
    data.to_csv(path + ".csv")


para1 = [-(3 ** (-0.5)), 3 ** (-1), -(3 ** (-1.5))]

para2 = [-0.5, 10 ** (-0.5), -(24 ** (-0.5))]

para3 = [-(3 ** (-0.5)), 7 ** (-0.5), -(17 ** (-0.5))]

paras = [para1, para2, para3]
funcs = [
    expected_cut_fraction_1local,
    expected_cut_fraction_2local,
    expected_cut_fraction_3local,
]
# start thread
for i, func in enumerate(funcs):
    for j, para in enumerate(paras):
        path = "graph2_output/" + str(i + 1) + "local_para" + str(j + 1) + "_10m"
        thread = Process(target=func, args=(path, para, 50))
        thread.start()

# Thread(target=expected_cut_fraction_3local,args=('out.txt', paras[2], 50))
