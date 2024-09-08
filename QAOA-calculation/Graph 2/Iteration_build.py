import numpy as np
from math import pi, cos, sin
import sys

# define the objective function
def objective(x,p,pre_E,pre_H1,pre_r1):
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
    E = np.einsum("m,ijklm->ijkl", Gamma, pre_E)
    E = np.exp(-1j * E)
    E_1 = np.einsum("m,ijm->ij", Gamma, pre_H1)
    E_1 = np.exp(-1j * E_1)
    E_r1 = np.einsum("m,ijklm->ijkl", Gamma, pre_r1)
    E_r1 = np.exp(-1j * E_r1)
    # initialize H[1]
    H[1]=np.einsum("b,ab->a", f_pre, E_1)**2

    # calculation of H
    for i in range(p-1):
        H[i + 2] = np.einsum(
            "b,c,d,b,d,c,abcd->a", f_pre, f_pre, f_pre, H[i+1], H[i+1], H[i], E)
    if p==1:
        res = np.einsum("a,b,a,b,c,d,a,b,c,d,abcd->", a[:,p], a[:,p], f_pre, f_pre, f_pre, f_pre, H[p], H[p], H[p-1], H[p-1], E_r1)
    else:
        res=np.einsum("a,b,a,b,c,d,a,b,c,d,abcd->", a[:,p], a[:,p], f_pre, f_pre, f_pre, f_pre, H[p], H[p], H[p-1], H[p-1], E)

    return 0.5 * res.real

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