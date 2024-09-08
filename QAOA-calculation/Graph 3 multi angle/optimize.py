from scipy.optimize import minimize
from math import pi
import os
import numpy as np
from Iteration_build import *

# called after each iteration
def callback(intermediate_result):
    global finish
    finish+=1
    with open(out_file, "a") as f:
        f.write("Iterations %d: " %(finish,))
        f.write("gamma, beta: %s, %s, " %(str(intermediate_result.x[0:2*p]), str(intermediate_result.x[2*p:])))
        f.write("cut fraction: %s\n" %(str(0.5-intermediate_result.fun),))
        f.flush()

if __name__ == '__main__':
    #----------------------------------global parameters and path------------------------------------------
    p = 3
    out_file = "Graph 3 copy/3.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    # ---------------------------------initial guess-------------------------------------------------------
    # p = 1
    # gamma=np.array([[0.2851,0.2851],[0.6,0.6],[pi/2,pi/4],[pi/2,0],[pi/4,pi/4],[pi/6,pi/6],[pi/8,pi/6],[0,pi/4],[pi/6,pi/6],[pi/8,pi/8],[0,0],[0,pi/3]])
    # beta=np.array([[0.3481],[pi/8],[pi/2],[pi/4],[pi/2],[pi/8],[pi/8],[pi/4],[pi/6],[0],   [0],[pi/3]])

    # p = 2
    # gamma = np.array([[0.2917,0.5623,0.2917,0.5623],[0.3817, 0.6655,0.3817,0.6655],[pi/2,0,pi/2,0],[pi/4,pi/6,pi/4,pi/4],[pi/8,pi/6,pi/8,0],[pi/8,pi/4,pi/6,pi/2],[pi/8,0,0,pi/6]])  
    # beta = np.array([[0.4090,0.2408],[0.4960, 0.2690], [pi/2,pi/2],[pi/4,pi/4],[pi/8,pi/8],[0,pi/3],[0,pi/4]])  
    # gamma = np.array([[0.19088048,0,0.50212489,0]])
    # beta = np.array([[0.38359873,0]])

    # # p = 3
    gamma = np.array([[0,1.32566731,1.57080114,0.78538862,1.36150781,0.34150466],[0.2014,0.48550,0.5916,0.2014,0.48550,0.5916],[0.3297, 0.5688, 0.6406,0.3297, 0.5688, 0.6406],[pi/8,pi/8,pi/8,pi/8,pi/8,pi/8],[pi/6,0,pi/6,0,pi/6,0],[pi/4,pi/3,pi/3,pi/4,pi/3,pi/3],[pi/8,pi/8,pi/8,0,0,0]])  
    beta =  np.array([[0.39269917,0.78539455,0.78541079],[0.5410,0.3187,0.1851],[0.5500, 0.3675, 0.2109],[pi/8,pi/8,pi/8],[pi/6,pi/6,pi/6],[pi/8,pi/8,pi/8],[pi/3,pi/4,pi/8]]) 

    # rand_gamma = np.array([np.multiply(np.random.random(2*p), pi/2) for i in range(20)])
    # rand_beta = np.array([np.multiply(np.random.random(p), pi/2) for i in range(20)])
    # # gamma = np.concatenate((gamma,rand_gamma))
    # beta = np.concatenate((beta,rand_beta))
    #-----------------------------------preprocessing------------------------------------------------------
    indices = np.arange(1 << (2 * p + 1))
    a = np.array([idx2arr(idx,p) for idx in indices]).astype(dtype=np.int8)
    # Perform element-wise multiplication and addition using vectorized operations
    # (ab+bc+cd+db)
    # pre_H = a[:, None, None, None] * a[:, None, None] + a[:, None, None] * a[:, None] +  a[:, None] * a + a * a[:, None, None]
    pre_E_vec = a[:,None] * a #ab
    # pre_H2 = a[:, None, None, None] * a[:, None, None] + a[:, None, None] * a[:, None] + a * a[:, None, None] #ab+bc+db
    pre_E_ring = a[:, None, None] * a[:, None] + a[:, None] * a +  a * a[:, None, None]
    print("pre-calculations done!")

    #------------------------------------output----------------------------------------------------------
    with open(out_file, "a") as f:
        f.write("layers p = %d\n\n" % (p))
        f.flush()
        for gamma0, beta0 in zip(gamma, beta):
            # initial points
            x0 = np.hstack((gamma0, beta0))
            # bounds on gamma and beta
            bounds = [[0, 2 * pi]] * (2*p) + [[0, pi]] * p

            f.write("Initially:    ")
            f.write("gamma, beta: %s, %s, " %(str(gamma0),str(beta0)))
            f.write("cut fraction: %s\n" %(str(0.5-objective(x0,p,pre_E_ring,pre_E_vec)),))
            f.flush()

            # optimize the objective function
            finish = 0
            res = minimize(objective, x0, args=(p,pre_E_ring,pre_E_vec), bounds=bounds,callback=callback)
            
            f.write("Success or not: %s\n" %(str(res.success),))
            f.write("Reasons for stopping: %s\n\n" %(res.message,))
            f.flush()
