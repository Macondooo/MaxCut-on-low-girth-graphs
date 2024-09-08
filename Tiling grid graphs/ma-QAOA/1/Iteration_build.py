import numpy as np
from math import pi, cos, sin
import sys


# define the objective function
def objective(x, p, pre_E):
    print("called")
    gamma1 = x[0:p]
    gamma2 = x[p:2*p]
    beta = x[2*p:]
    Gamma1 = np.hstack((gamma1, [0], -gamma1[::-1]))
    Gamma2 = np.hstack((gamma2, [0], -gamma2[::-1]))
    a = np.zeros((1 << (2 * p + 1), 2 * p + 1))

    # pre-calculation of f(a)
    f_pre = np.zeros(1 << (2 * p + 1), dtype=complex)
    for idx in range(1 << (2 * p + 1)):
        a[idx] = idx2arr(idx, p)
        f_pre[idx] = func(a[idx], beta, p)

    E1 = np.einsum("m,ijm->ij", Gamma1, pre_E)
    E1 = np.exp(-1j * E1)
    E2 = np.einsum("m,ijm->ij", Gamma2, pre_E)
    E2 = np.exp(-1j * E2)

    # 1-local
    if p ==1:
        f1 = np.einsum("b,c,ab,ac->a",f_pre,f_pre,E2,E1)
        res1 = np.einsum("a,b,a,b,a,b,ab->", a[:, p], a[:, p], f_pre, f_pre, f1, f1, E1)
        
        f2 = np.einsum("b,ab->a",f_pre,E1)**2
        res2 = np.einsum("a,b,a,b,a,b,ab->", a[:, p], a[:, p], f_pre, f_pre, f2, f2, E2)
        return 0.5 * (2/3 * res1.real + 1/3 * res2.real)

    # 2-local
    if p == 2:
        # edge 1
        Y1 = np.einsum("b,c,d,cb,db,ba->a",f_pre,f_pre,f_pre,E1,E1,E2)
        tmp1 = np.einsum("b,ab->a",f_pre,E2)
        res1 = np.einsum("a,b,a,b,c,d,e,a,b,e,c,ab,bc,cd,de,ea->",a[:, p],a[:, p],f_pre,f_pre,f_pre,f_pre,f_pre,Y1,Y1, tmp1,tmp1,E1,E1,E1,E1,E1)

        # edge 2
        Y2= np.einsum("b,c,d,cb,db,ba->a",f_pre,f_pre,f_pre,E1,E2,E1)**2
        res2 = np.einsum("a,b,a,b,a,b,ab->", a[:, p], a[:, p], f_pre, f_pre, Y2, Y2, E2)

        return 0.5 * (2/3 * res1.real+1/3 * res2.real)


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
