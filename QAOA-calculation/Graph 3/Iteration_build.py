import numpy as np
from math import pi, cos, sin
import sys

# define the objective function
def objective(x,p,pre_H,pre_H1,pre_r1):
    print("called")
    gamma = x[0:p]
    beta = x[p:]
    Gamma = np.hstack((gamma, [0], -gamma[::-1]))
    a = np.zeros((1 << (2 * p + 1), 2 * p + 1))

    # the iteration symbol, with initializing H[0,a] to 1
    # the configuration basis a, is numerically represented in lexicographic order
    H = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)

    # pre-calculation of f(a)
    f_pre = np.zeros(1 << (2 * p + 1), dtype=complex)
    for idx in range(1 << (2 * p + 1)):
        a[idx] = idx2arr(idx,p)
        f_pre[idx] = func(a[idx], beta,p)

    # pre-calculation of E
    E_H = np.einsum("m,ijklm->ijkl", Gamma, pre_H)
    E_H = np.exp(-1j * E_H)
    E_1 = np.einsum("m,ijm->ij", Gamma, pre_H1)
    E_1 = np.exp(-1j * E_1)
    # E_2 = np.einsum("m,ijklm->ijkl", Gamma, pre_H2)
    # E_2 = np.exp(-1j * E_2)
    E_r1 = np.einsum("m,ijkm->ijk", Gamma, pre_r1)
    E_r1 = np.exp(-1j * E_r1)

    # initialize H[1], H[2]
    H[1]=np.einsum("b,ab->a", f_pre, E_1)
    for i in range(p-1):
        H[i + 2] = np.einsum(
            "b,c,d,c,d,abcd->a", f_pre, f_pre, f_pre, H[i], H[i], E_H)
    # edge 1
    res1 = np.einsum("a,b,a,b,c,a,b,c,abc->", a[:,p], a[:,p], f_pre, f_pre, f_pre, H[p], H[p], H[p-1], E_r1)
    # edge 2
    G = np.einsum("b,c,b,c,abc->a",f_pre,f_pre,H[p-1],H[p-1],E_r1)
    res2 = np.einsum("a,b,a,b,a,b,ab->", a[:,p], a[:,p], f_pre, f_pre, G, G, E_1)

    

    return 1.0/3.0 * res1.real + 0.5/3.0 * res2.real

# convert an integer to (2p+1) bin array
def idx2arr(idx,p):
    a = np.zeros(2 * p + 1)
    for i in range(2 * p + 1):
        tmp = idx % 2
        if tmp == 0:
            a[i] = 1
        else: a[i] = -1
        idx = idx // 2

    return a

def func(a, beta,p):
    # a: np.array of size (2p+1), (a_1,a_2,...,a_p,a_0,a_{-p},...,a_{-1})
    # beta: np.array of size (p), (beta_1,beta_2,...,beta_p)
    res = 0.5
    for i in range(p):
        if a[i] == a[i + 1]:
            res = res * cos(beta[i])
        else:
            res = res * complex(0, sin(beta[i]))

    for i in range(p, 2 * p):
        if a[i] == a[i + 1]:
            res = res * cos(-beta[2 * p - i - 1])
        else:
            res = res * complex(0, sin(-beta[2 * p - i - 1]))

    return res