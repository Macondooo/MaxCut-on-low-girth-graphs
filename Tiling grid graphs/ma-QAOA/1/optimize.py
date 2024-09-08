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
    p = 2
    out_file = "qaoa-multi/1/p2.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    # ---------------------------------initial guess-------------------------------------------------------
    # p = 1
    # gamma=np.array([[0.30773985,0.30773985],[0.6,0.6],[pi/2,0],[0,pi/2],[pi/4,pi/8],[pi/6,0],[pi/8,pi/8],[pi/4,pi/4],[pi/6,pi/9],[pi/8,pi/3],[0,0],[pi/3,0]])
    # beta=np.array([[0.39269908],[pi/8],[pi/2],[pi/4],[pi/2],[pi/8],[pi/8],[pi/4],[pi/6],[0],   [0],[pi/3]])

    # p = 2
    gamma = np.array([[0.24306752,0.44653574,0.24306752,0.44653574],[0.30773982,0,0.30773933,0]])  
    beta = np.array([[0.54015487,0.27849337],[0.39269907,0]])  

    
    rand_gamma = np.array([np.multiply(np.random.random(2*p), pi/2) for i in range(50)])
    rand_beta = np.array([np.multiply(np.random.random(p), pi/2) for i in range(50)])
    gamma = np.concatenate((gamma,rand_gamma))
    beta = np.concatenate((beta,rand_beta))  

    #-----------------------------------preprocessing------------------------------------------------------
    indices = np.arange(1 << (2 * p + 1))
    a = np.array([idx2arr(idx,p) for idx in indices]).astype(dtype=np.int8)
    # Perform element-wise multiplication and addition using vectorized operations
    pre_E = a[:, None] * a
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
            f.write("cut fraction: %s\n" %(str(0.5-objective(x0,p,pre_E)),))
            f.flush()

            # optimize the objective function
            finish = 0
            res = minimize(objective, x0, args=(p,pre_E), bounds=bounds,callback=callback)
            
            f.write("Success or not: %s\n" %(str(res.success),))
            f.write("Reasons for stopping: %s\n\n" %(res.message,))
            f.flush()
