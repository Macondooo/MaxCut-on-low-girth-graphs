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
    out_file = "Graph 2 copy/3.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    # ---------------------------------initial guess-------------------------------------------------------
    # p = 1
    # gamma=np.array([[0.6,0],[pi/2,pi/2],[pi/2,pi/4],[pi/4,pi/4],[pi/6,pi/8],[pi/8,pi/2],[pi/4,pi/4],[pi/6,pi/3],[pi/8,pi/8],[0,0],[pi/3,0],[0,pi/2],[0,0.6]])
    # beta=np.array([[pi/8],[pi/2],[pi/4],[pi/2],[pi/8],[pi/8],[pi/4],[pi/6],[0],[0],[pi/3],[pi/4],[pi/8]])


    # p = 2

    
    # gamma = np.array([[0.19783633,0.35348299,0.19783633,0.35348299],[0.3817, 0.3817,0.6655,0.6655],[pi/2,pi/4,pi/2,0],[pi/6,pi/4,pi/3,pi/4],[pi/8,pi/8,pi/8,pi/8],[pi/8,pi/6,pi/6,0],[pi/6,pi/8,0,pi/6],[pi/6,pi/10,pi,pi/5],[pi/5,pi/8,pi/2,pi/5],[0,0,pi/10,0],[pi/4,pi/7,pi/9,pi/4]])  
    # beta = np.array([[0.55578159,0.31330863],[0.4960, 0.2690], [pi/2,pi/2],[pi/4,pi/4],[pi/8,pi/8],[0,pi/3],[0,pi/4],[pi/5,0],[pi/6,pi/3],[pi/10,0],[pi/3,pi/10]])
    # gamma = np.array([[0.26179935,0,0.26179957,0]])
    # beta = np.array([[0.39269908,0]])
    
    # gamma = np.array([[0.3817, 0.3817,0.6655,0.6655],[pi+pi/2,pi+pi/2],[pi/4,pi+pi/4],[pi+pi/8,pi+pi/8],[pi/8,pi/6],[pi+pi/8,pi+pi/6],[pi/10,pi/5],[pi/8,pi+pi/5],[pi/10,0],[pi+pi/7,pi+pi/9]])  
    # beta = np.array([[0.4960, 0.2690], [pi+pi/2,pi+pi/2],[pi+pi/4,pi/4],[pi+pi/8,pi/8],[0,pi+pi/3],[pi+0,pi+pi/4],[pi/5,0],[pi+pi/6,pi+pi/3],[pi+pi/10,0],[pi+pi/3,pi+pi/10]])  

    # # p = 3
    gamma = np.array([[0.3297,0.5688,0.6406,0.3297,0.5688,0.6406],[pi/8,pi/8,pi/8,0,pi/6,pi/2],[pi/6,pi/6,pi/6,pi/4,pi/3,0],[pi/4,pi/3,pi/3,0,0,0],[pi/8,pi/8,0,pi/8,pi/6,pi/8]])  
    beta =  np.array([[0.5500, 0.3675, 0.2109],[pi/8,pi/8,pi/8],[pi/6,pi/6,pi/6],[pi/8,pi/8,pi/8],[pi/3,pi/4,pi/8]])  
    
    rand_gamma = np.array([np.multiply(np.random.random(2*p), pi/2) for i in range(50)])
    rand_beta = np.array([np.multiply(np.random.random(p), pi/2) for i in range(50)])
    gamma = np.concatenate((gamma,rand_gamma))
    beta = np.concatenate((beta,rand_beta))

    #-----------------------------------preprocessing------------------------------------------------------
    indices = np.arange(1 << (2 * p + 1))
    a = np.array([idx2arr(idx,p) for idx in indices]).astype(dtype=np.int8)
    # Perform element-wise multiplication and addition using vectorized operations
    # (ab+bc+cd+da)
    pre_H1 = a[:,None] * a
    if p == 1:
        pre_r1 = a[:, None, None, None] * a[:, None, None] + a[:, None, None] * a[:, None] + a * a[:, None, None, None]
    else: pre_r1 = None
    # ab+bc+da
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
            f.write("cut fraction: %s\n" %(str(0.5-objective(x0,p,pre_H1,pre_r1)),))
            f.flush()

            # optimize the objective function
            finish = 0
            res = minimize(objective, x0,method="COBYLA",args=(p,pre_H1,pre_r1), bounds=bounds,callback=callback)
            
            f.write("Success or not: %s\n" %(str(res.success),))
            f.write("Reasons for stopping: %s\n\n" %(res.message,))
            f.flush()
