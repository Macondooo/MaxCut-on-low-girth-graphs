import numpy as np
from math import pi, cos, sin
import sys


def func(a, beta, p):
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


# convert an integer to (2p+1) bin array
def idx2arr(idx, p):
    a = np.zeros(2 * p + 1)
    for i in range(2 * p + 1):
        tmp = idx % 2
        if tmp == 0:
            a[i] = 1
        else:
            a[i] = -1
        idx = idx // 2

    return a


# define the objective function
# def cal_HG(x,p,pre_E_G,pre_E_H):
#     gamma = x[0:p]
#     beta = x[p:]
#     Gamma = np.hstack((gamma, [0], -gamma[::-1]))

#     # the iteration symbol, with initializing G[0,a] and H[0,a] to 1
#     # the configuration basis a, is numerically represented in lexicographic order
#     G = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)
#     H = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)

#     # pre-calculation of f(a)
#     f_pre = np.zeros(1 << (2 * p + 1), dtype=complex)
#     a = np.zeros((1 << (2 * p + 1), 2 * p + 1))
#     for idx in range(1 << (2 * p + 1)):
#         a[idx] = idx2arr(idx,p)
#         f_pre[idx] = func(a[idx], beta, p)

#     # pre-calculation of E_G, E_H
#     E_G = np.einsum("k,ijk->ij", Gamma, pre_E_G)
#     E_G = np.exp(-1j * E_G)
#     E_H = np.einsum("m,ijklm->ijkl", Gamma, pre_E_H)
#     E_H = np.exp(-1j * E_H)

#     # calculation of G and H
#     for i in range(p):
#         # calculation of G
#         G[i + 1] = np.einsum("i,i,ji->j", f_pre, H[i], E_G) ** 2
#         # calculation of H
#         H[i + 1] = np.einsum(
#             "j,k,l,j,k,l,ijkl->i", f_pre, f_pre, f_pre, G[i], G[i], H[i], E_H
#         )

#     return G, H


def objective(x, p, pre_E, pre_E_ring):
    print("called")
    gamma = x[0:p]
    beta = x[p :]
    Gamma = np.hstack((gamma, [0], -gamma[::-1]))

    # the iteration symbol, with initializing G[0,a] and H[0,a] to 1
    # the configuration basis a, is numerically represented in lexicographic order
    G = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)
    H = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)

    # pre-calculation of f(a)
    f_pre = np.zeros(1 << (2 * p + 1), dtype=complex)
    a = np.zeros((1 << (2 * p + 1), 2 * p + 1))
    for idx in range(1 << (2 * p + 1)):
        a[idx] = idx2arr(idx, p)
        f_pre[idx] = func(a[idx], beta, p)

    # pre-calculation
    E = np.einsum("k,ijk->ij", Gamma, pre_E)
    E = np.exp(-1j * E)

    E_ring = np.einsum("m,ijkm->ijk", Gamma, pre_E_ring)
    E_ring = np.exp(-1j * E_ring)


    # calculation of G and H
    G[1] = np.einsum("b,ab->a", f_pre, E)**2
    H[1] = np.einsum("b,ab->a", f_pre, E)**3
    if p == 2 or p == 3:
        G[2] = np.einsum("b,b,ab->a",H[1],f_pre,E)**2
        tmp = np.einsum("b,b,ab->a",H[1],f_pre,E)
        H[2] = np.einsum("b,c,b,c,a,abc->a", f_pre, f_pre, G[1], G[1], tmp, E_ring)
        if p == 3: 
            G[3] = np.einsum("b,b,ab->a", f_pre, H[2], E) ** 2
            tmp3  = np.einsum("b,b,ab->a", f_pre, H[2], E)
            H[3] = np.einsum("b,c,b,c,a,abc->a", f_pre, f_pre, G[2], G[2], tmp3, E_ring)


    # first kind of edge
    res1 = np.einsum(
        "i,j,i,j,k,i,j,k,ijk->",a[:, p],a[:, p],f_pre,f_pre,f_pre,G[p],G[p],G[p - 1],E_ring,
    )
    # second kind of edge
    res2 = np.einsum(
        "i,j,i,j,i,j,ij->", a[:, p], a[:, p], f_pre, f_pre, H[p], H[p], E)

    return 0.5 * (0.5 * res1.real + 0.5 * res2.real)
