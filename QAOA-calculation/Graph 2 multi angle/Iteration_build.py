import numpy as np
from math import pi, cos, sin
import sys

# define the objective function
def objective(x,p,pre_H1,pre_r1):
    print("called")
    gamma1 = x[0:p]
    gamma2 = x[p:2*p]
    beta = x[2*p:]
    Gamma1 = np.hstack((gamma1, [0], -gamma1[::-1]))
    Gamma2 = np.hstack((gamma2, [0], -gamma2[::-1]))
    
    a = np.zeros((1 << (2 * p + 1), 2 * p + 1))

    # the iteration symbol, with initializing H[0,a] to 1
    # the configuration basis a, is numerically represented in lexicographic order
    H1 = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)
    H2 = np.full((p + 1, 1 << (2 * p + 1)), 1, dtype=complex)

    # pre-calculation of f(a)
    f_pre = np.zeros(1 << (2 * p + 1), dtype=complex)
    for idx in range(1 << (2 * p + 1)):
        a[idx] = idx2arr(idx,p)
        f_pre[idx] = func(a[idx], beta,p)

    # pre-calculation of E
    # E_ring1 = np.einsum("m,ijklm->ijkl", Gamma1, pre_E)
    # E_ring2 = np.einsum("m,ijklm->ijkl", Gamma2, pre_E)
    # E_ring1 = np.exp(-1j * E_ring1)
    # E_ring2 = np.exp(-1j * E_ring2)
    
    E_1 = np.einsum("m,ijm->ij", Gamma1, pre_H1)
    E_1 = np.exp(-1j * E_1)
    E_2 = np.einsum("m,ijm->ij", Gamma2, pre_H1)
    E_2 = np.exp(-1j * E_2)
    
    if p == 1:
        E_r1 = np.einsum("m,ijklm->ijkl", Gamma1, pre_r1)
        E_r1 = np.exp(-1j * E_r1)
        E_r2 = np.einsum("m,ijklm->ijkl", Gamma2, pre_r1)
        E_r2 = np.exp(-1j * E_r2)
    
    # initialize H[1]
    H1[1]=np.einsum("b,ab->a", f_pre, E_1)**2
    H2[1]=np.einsum("b,ab->a", f_pre, E_2)**2
    if p == 2 or p == 3:
        H2[2] = np.einsum("b,c,d,b,d,ab,bc,cd,da->a",f_pre,f_pre,f_pre,H1[1],H1[1],E_2,E_2,E_2,E_2)
        H1[2] = np.einsum("b,c,d,b,d,ab,bc,cd,da->a",f_pre,f_pre,f_pre,H2[1],H2[1],E_1,E_1,E_1,E_1)
        if p == 3:
            H1[3] = np.einsum("b,c,d,b,c,d,ab,bc,cd,da->a",f_pre,f_pre,f_pre,H2[2],H2[1],H2[2],E_1,E_1,E_1,E_1)
            H2[3] = np.einsum("b,c,d,b,c,d,ab,bc,cd,da->a",f_pre,f_pre,f_pre,H1[2],H1[1],H1[2],E_2,E_2,E_2,E_2)
    # calculation of H
    # for i in range(p-1):
        # H1[i + 2] = np.einsum(
        #     "b,c,d,b,d,c,abcd->a", f_pre, f_pre, f_pre, H2[i+1], H2[i+1], H2[i], E_ring1)
        # H2[i + 2] = np.einsum(
        #     "b,c,d,b,d,c,abcd->a", f_pre, f_pre, f_pre, H1[i+1], H1[i+1], H1[i], E_ring2)
    if p==1:
        res1 = np.einsum("a,b,a,b,c,d,a,b,abcd->", a[:,p], a[:,p], f_pre, f_pre, f_pre, f_pre, H2[p], H2[p], E_r1)
        res2 = np.einsum("a,b,a,b,c,d,a,b,abcd->", a[:,p], a[:,p], f_pre, f_pre, f_pre, f_pre, H1[p], H1[p], E_r2)
    else:
        res1 = np.einsum("a,b,a,b,c,d,a,b,c,d,ab,bc,cd,da->", a[:,p], a[:,p], f_pre, f_pre, f_pre, f_pre, H2[p], H2[p], H2[p-1], H2[p-1],E_1,E_1,E_1,E_1)
        res2 = np.einsum("a,b,a,b,c,d,a,b,c,d,ab,bc,cd,da->", a[:,p], a[:,p], f_pre, f_pre, f_pre, f_pre, H1[p], H1[p], H1[p-1], H1[p-1], E_2,E_2,E_2,E_2)

    return 0.5 * (0.5*res1.real + 0.5*res2.real)

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