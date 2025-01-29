__Author__ = "Amina Benaceur(*)"
__Copyright__ = "November 2024 \
                   (*)MIT"

#import time
import os
import time
import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
from greedy_utils import read_vector, strong_greedy, errPlot, predictive_greedy, rel_POD, read_matrix, get_error_decay
dir_path = "/Users/amina/Desktop/um6p_greedy/code/helmoltz/"
FEM_DIR = dir_path + "Offline/fem_sols/"
MAT_DIR = dir_path + "Offline/matrices"

#start = time.time()
nb_par = 50
threshold = 1e-11
#aa = os.system("freefem++ -v 0 "+dir_path+"helmoltz.edp")

nb_dof, _ = read_vector(FEM_DIR, "sol_1.txt")


"""
KK = read_matrix(MAT_DIR, "/stiff.txt", nb_dof)

MM = read_matrix(MAT_DIR, "/mass.txt", nb_dof)

scale = 1/8.
SPM = KK + scale*MM

Xoff = np.matrix(np.zeros((nb_par, nb_dof)))
for jaux in range(nb_par):
    _, sol = read_vector(FEM_DIR, "sol_"+str(jaux+1)+".txt")
    Xoff[jaux,:] = sol.T
#"""
Xoff = np.matrix(np.random.randn(nb_par, nb_dof))

SPM = sp.sparse.eye(nb_dof, format = "csc")

maxModes = nb_par


start = time.time()
pr_selec_params, pr_err = predictive_greedy(Xoff, SPM, threshold, nb_par, maxModes)
Tpg = time.time()-start

start = time.time()

st_selec_params, st_err = strong_greedy(Xoff, SPM, threshold, nb_par, maxModes)
Tsg = time.time()-start


start = time.time()
_, sg_vals, _ = rel_POD(Xoff.T, SPM, threshold, maxModes)
Tpod = time.time()-start
print("-----------------------------------")



print("----------- predictive greedy -------------")
pr_errors = get_error_decay(Xoff, pr_selec_params, maxModes, SPM)
print("-----------------------------------")

print("----------- strong greedy -------------")
st_errors = get_error_decay(Xoff, st_selec_params, maxModes, SPM)
print("-----------------------------------")


errPlot(np.arange(maxModes)+1, sg_vals[0:maxModes], "iteration", "Error", 1, 1, "*", "b", "POD", "Heat equation")
errPlot(np.arange(len(st_errors))+1, st_errors, "iteration", "Error", 1, 1, "*", "k", "Strong greedy", "Heat equation")
errPlot(np.arange(len(pr_errors))+1, pr_errors, "iteration", "Error", 1, 1, ">", "r", "Predictive greedy", "Heat equation")

print "POD : ", Tpod, "s"
print "strong greedy : ", Tsg, "s"
print "predictive greedy : ", Tpg, "s"
