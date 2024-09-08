import numpy as np
from math import pi, cos, sin
import sys


# define the objective function
def objective(x, p, v2, v3):
    print("called")
    gamma = x[0:p]
    beta = x[p:]
    Gamma = np.hstack((gamma, [0], -gamma[::-1]))
    a = np.zeros((1 << (2 * p + 1), 2 * p + 1))

    # pre-calculation of f(a)
    f_pre = np.zeros(1 << (2 * p + 1), dtype=complex)
    for idx in range(1 << (2 * p + 1)):
        a[idx] = idx2arr(idx, p)
        f_pre[idx] = func(a[idx], beta, p)

    E2 = np.einsum("m,ijm->ij", Gamma, v2)
    E2 = np.exp(-1j * E2)
    E3 = np.einsum("m,ijkm->ijk", Gamma, v3)
    E3 = np.exp(-1j * E3)

    # 1-local
    if p == 1:
        H1 = np.einsum("b,ab->a", f_pre, E2) ** 3
        res1 = np.einsum("a,b,a,b,a,b,ab->", a[:, p], a[:, p], f_pre, f_pre, H1, H1, E2)

        G1 = np.einsum("b,ab->a", f_pre, E2) ** 2
        res2 = np.einsum(
            "a,b,a,b,c,a,b,abc->", a[:, p], a[:, p], f_pre, f_pre, f_pre, G1, G1, E3
        )

        return 0.5 * (0.5 * res1.real + 0.5 * res2.real)

    # 2-local
    if p == 2:
        H2 = np.einsum("b,ab->a", f_pre, E2) ** 2
        H1 = np.einsum("b,ab->a", f_pre, E2)
        # H_4_9_0 = np.einsum("a,a,abc->bc", f_pre, H2, E_ab_ac)
        # H_0_2 = np.einsum("a,a,abcd",f_pre,H1,E_ab_ac_ad,)
        sum1 = np.einsum(
            "i,j,h,h,j,ih,ij,ja,ha,gh->ga",
            f_pre,
            f_pre,
            f_pre,
            H1,
            H2,
            E2,
            E2,
            E2,
            E2,
            E2
        )
        res1 = np.einsum(
            "ga,fb,g,a,f,b,g,f,a,b,gf,ga,ab,bf",
            sum1,
            sum1,
            f_pre,
            f_pre,
            f_pre,
            f_pre,
            H1,
            H1,
            a[:, p],
            a[:, p],
            E2,
            E2,
            E2,
            E2,
        )

        sum2_1 = np.einsum(
            "h,i,j,h,j,hi,ij,hj,ah,bj->ab",
            f_pre,
            f_pre,
            f_pre,
            H1,
            H1,
            E2,
            E2,
            E2,
            E2,
            E2,
        )
        sum2_2 = np.einsum("g,f,g,gf,fe,ga->ae", f_pre, f_pre, H2, E2, E2, E2)
        res2 = np.einsum(
            "a,b,e,a,b,ae,be,ab,ae,eb,ab",
            f_pre,
            f_pre,
            f_pre,
            a[:, p],
            a[:, p],
            sum2_2,
            sum2_2,
            sum2_1,
            E2,
            E2,
            E2,
        )


        return 0.5 * (0.5 * res1.real + 0.5 * res2.real)


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
