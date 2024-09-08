from scipy.optimize import minimize
from math import pi
import os
import numpy as np
from Iteration_build import *


# def callback(intermediate_result):
#     global finish
#     finish+=1
#     with open(out_file, "a") as f:
#         f.write("Iterations %d: " %(finish,))
#         f.write("gamma, beta: %s, %s, " %(str(intermediate_result.x[0:p]), str(intermediate_result.x[p:])))
#         f.write("cut fraction: %s\n" %(str(0.5-intermediate_result.fun),))
#         G, H = cal_HG(intermediate_result.x)
#         print("G",file=f)
#         print(G,file=f)
#         print("H",file=f)
#         print(H,file=f)
#         f.flush()
def callback(intermediate_result):
    global finish
    finish+=1
    with open(out_file, "a") as f:
        f.write("Iterations %d: " %(finish,))
        f.write("gamma, beta: %s, %s, " %(str(intermediate_result.x[0:2*p]), str(intermediate_result.x[2*p:])))
        f.write("cut fraction: %s\n" %(str(0.5-intermediate_result.fun),))
        f.flush()
    
    # percentage = round( progress_bar.finish_tasks_number /  1000 * 100)
    # print("\rprogress: {}%: ".format(percentage), " " * (percentage // 2), end="")
    # sys.stdout.flush()



#----------------------------------global parameters and path------------------------------------------
p = 3
out_file = "Graph 1 copy/3.txt"
if os.path.exists(out_file):
	os.remove(out_file)
# ---------------------------------initial points-------------------------------------------------------
# p = 1
# gamma=np.array([[0.19012958,0.31953401],[0.25367792,0.25367792],[0.6,0.6],[pi/2,pi/4],[pi/2,pi/2],[pi/4,pi/6],[pi/6,pi/6],[pi/8,pi/10],[pi/4,pi/4],[pi/6,pi/8],[pi/8,pi/8],[0,0],[0,pi/3]])
# beta=np.array([[0.38144808],[0.36625718],[pi/8],[pi/2],[pi/4],[pi/2],[pi/8],[pi/8],[pi/4],[pi/6],[0],   [0],[pi/3]])
# gamma=np.array([[0.25367792,0.25367792],[0.6,0.6],[pi/2,pi/4],[pi/2,pi/2],[pi/4,pi/6],[pi/6,pi/6],[pi/8,pi/10],[pi/4,pi/4],[pi/6,pi/8],[pi/8,pi/8],[0,0],[0,pi/3]])
# beta=np.array([[0.36625718],[pi/8],[pi/2],[pi/4],[pi/2],[pi/8],[pi/8],[pi/4],[pi/6],[0],   [0],[pi/3]])

# p = 2
# gamma = np.array([[0,0,0.78540024,0],[0.20482498,0.39019961,0.20482498,0.39019961],[0.3817, 0.6655, 0.3817, 0.6655]])  
# beta = np.array([[0.39269922,0],[0.48577091,0.25781431],[0.4960, 0.2690]])  
# gamma = np.array([[0,0,0.78540024,0]])
# beta = np.array([[0.39269922,0]])

# p = 3
# gamma = np.array([[0.3297, 0.5688, 0.6406],[pi/8,pi/8,pi/8],[pi/6,pi/6,pi/6],[pi/4,pi/3,pi/3],[pi/8,pi/8,pi/8]])  
# beta =  np.array([[0.5500, 0.3675, 0.2109],[pi/8,pi/8,pi/8],[pi/6,pi/6,pi/6],[pi/8,pi/8,pi/8],[pi/3,pi/4,pi/8]])
gamma = np.array([[0.17868793,0.3448119,0.40006796,0.17868793,0.3448119,0.40006796,]])  
beta =  np.array([[0.55063304,0.37389559,0.20167476]])  

rand_gamma = np.array([np.multiply(np.random.random(2*p), pi/2) for i in range(50)])
rand_beta = np.array([np.multiply(np.random.random(p), pi/2) for i in range(50)])
gamma = np.concatenate((gamma,rand_gamma))
beta = np.concatenate((beta,rand_beta))
    
#---------------------------------preprocessing---------------------------------------------------------
# Generate arrays a, b, c, d for all indices
indices = np.arange(1 << (2 * p + 1))
a = np.array([idx2arr(idx,p) for idx in indices]).astype(dtype=np.int8)
# Perform element-wise multiplication and addition using vectorized operations
pre_E = a[:, None] * a #ab
# pre_E_H = a[:, None, None, None] * a[:, None, None] + a[:, None, None, None] * a[:, None] +  a[:, None, None, None] * a + a[:, None, None] * a[:, None]#(ab+ac+ad+bc)
pre_E_ring = a[:, None, None] * a[:, None] + a[:, None, None] * a + a[:, None] * a #ab+ac+bc
print("pre-calculations done!")

#---------------------------------optimizing------------------------------------------------------------
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
        f.write("cut fraction: %s\n" %(str(0.5-objective(x0,p, pre_E, pre_E_ring)),))
        f.flush()

        # optimize the objective function
        finish = 0
        res = minimize(objective, x0, args=(p, pre_E, pre_E_ring), bounds=bounds,callback=callback)
        
        
        # output results
        # f.write("optimized gamma: %s\n" %(str(res.x[:p]),))
        # f.write("optimized beta: %s\n" %(str(res.x[p:]),))
        # f.write("optimized function: %s\n" %(str(0.5 - res.fun),))
        f.write("Success or not: %s\n" %(str(res.success),))
        f.write("Reasons for stopping: %s\n\n" %(res.message,))
        f.flush()
