import numpy as np
from math import pi, cos, sin
import sys

# define the objective function
def objective(x,p,pre_E_ring,pre_E_vec):
    print("called")
    gamma1 = x[0:p]
    gamma2 = x[p:2*p]
    beta = x[2*p:]
    Gamma1 = np.hstack((gamma1, [0], -gamma1[::-1]))
    Gamma2 = np.hstack((gamma2, [0], -gamma2[::-1]))
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
    E_ring = np.einsum("m,ijkm->ijk", Gamma1, pre_E_ring)
    E_ring = np.exp(-1j * E_ring)
    
    E_vec1 = np.einsum("m,ijm->ij", Gamma1, pre_E_vec)
    E_vec1 = np.exp(-1j * E_vec1)
    E_vec2 = np.einsum("m,ijm->ij", Gamma2, pre_E_vec)
    E_vec2 = np.exp(-1j * E_vec2)

    H[1] = np.einsum("b,ab->a", f_pre, E_vec2)
    if p == 2 or p == 3:
        tmp2 = np.einsum("b,ab->a", f_pre, E_vec1)**2
        H[2] = np.einsum("b,b,ab->a", f_pre, tmp2, E_vec2)
        if p == 3:
            tmp3 = np.einsum("b,c,b,c,abc->a", f_pre,f_pre,H[1],H[1],E_ring)
            H[3] = np.einsum(
                "b,b,ab->a",f_pre,tmp3,E_vec2)
    # edge 1
    res1 = np.einsum("a,b,a,b,c,a,b,c,abc->", a[:,p], a[:,p], f_pre, f_pre, f_pre, H[p], H[p], H[p-1], E_ring)
    # edge 2
    if p == 1:
        G = np.einsum("b,ab->a", f_pre, E_vec1)**2
    else:
        G = np.einsum("b,c,b,c,abc->a",f_pre,f_pre,H[p-1],H[p-1],E_ring)  
    res2 = np.einsum("a,b,a,b,a,b,ab->", a[:,p], a[:,p], f_pre, f_pre, G, G, E_vec2)

    

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