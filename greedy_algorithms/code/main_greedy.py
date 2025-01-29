__Author__ = "Amina Benaceur(*)"
__Copyright__ = "November 2024 \
                   (*)UM6P"

#import time
import os
import time
import numpy as np
import scipy as sp
from math import sqrt
from scipy.sparse import csc_matrix
from greedy_utils import read_vector, read_matrix
from greedy_utils import standard_greedy, predictive_greedy, reverse_greedy
from greedy_utils import errPlot, rel_POD, get_error_decay
dir_path = "/home/benaceur/Documents/greedy_algorithms/code/laplacian/"
FEM_DIR = dir_path + "Offline/fem_sols/"
MAT_DIR = dir_path + "Offline/matrices"

#start = time.time()
nb_par = 200
threshold = 1e-11
"""
#aa = os.system("FreeFem++ -v 0 "+dir_path+"laplacian.edp")
nb_dof, _ = read_vector(FEM_DIR, "/sol_1.txt")


KK = read_matrix(MAT_DIR, "/stiff.txt", nb_dof)

MM = read_matrix(MAT_DIR, "/mass.txt", nb_dof)

scale = 1/8.
SPM = KK + scale*MM

Xoff = np.matrix(np.zeros((nb_par, nb_dof)))
for jaux in range(nb_par):
    _, sol = read_vector(FEM_DIR, "sol_"+str(jaux+1)+".txt")
    Xoff[jaux,:] = sol.T
"""
nb_dof = 2000

Xoff = np.matrix(np.random.randn(nb_par, nb_dof))

SPM = sp.sparse.eye(nb_dof, format = "csc")

#"""
#Build a tailored set:
import numpy as np
# Values starting from 1, incremented by 0.01
diagonal_values = [1 + 0.01 * i for i in range(nb_dof)]
# Create the diagonal matrix
nb_generated_vecs = int(np.floor(nb_dof/4))
diag_matrix = np.diag(diagonal_values)[:nb_generated_vecs,:]

vec_w = np.matrix(np.zeros((int(np.floor((nb_generated_vecs-1)/2)),nb_dof)))
for kk in range(int(np.floor(nb_generated_vecs-1)/2)):
    vec_w[kk, 2*kk+1:2*kk+3] = .999*diagonal_values[2*kk]/sqrt(2)

Xoff = np.concatenate((diag_matrix, vec_w), axis = 0)

nb_par = min(Xoff.shape)
#"""

maxModes = int(np.floor(nb_par/50))


start = time.time()
pr_selec_params, pr_err = predictive_greedy(Xoff, SPM, threshold, nb_par, maxModes)
Tpred = time.time()-start

start = time.time()
st_selec_params, st_err = standard_greedy(Xoff, SPM, threshold, nb_par, maxModes)
Tstd = time.time()-start

#start = time.time()
#rev_selec_params, rev_err = reverse_greedy(Xoff.T, SPM, threshold, nb_par, maxModes)
#Trev = time.time()-start

start = time.time()
_, sg_vals, _ = rel_POD(Xoff.T, SPM, threshold, maxModes)
Tpod = time.time()-start

"""
pr_errors = get_error_decay(Xoff, pr_selec_params, maxModes, SPM)
st_errors = get_error_decay(Xoff, st_selec_params, maxModes, SPM)
#rev_errors = get_error_decay(Xoff, rev_selec_params, maxModes, SPM)

errPlot(np.arange(maxModes)+1, sg_vals[0:maxModes], "iteration", "Error", 1, 1, "*", "b", "POD", "Heat equation")
errPlot(np.arange(len(st_errors))+1, st_errors, "iteration", "Error", 1, 1, "s", "k", "Standard greedy", "Heat equation")
errPlot(np.arange(len(pr_errors))+1, pr_errors, "iteration", "Error", 1, 1, "*", "r", "Predictive greedy", "Heat equation")
#errPlot(np.arange(len(rev_errors))+1, rev_errors, "iteration", "Error", 1, 1, "*", "g", "Reverse greedy", "Heat equation")
#"""
errPlot(np.arange(len(st_err))+1, st_err, "iteration", "Error", 1, 1, "*", "g", "Standard greedy", "Heat equation")
errPlot(np.arange(len(pr_err))+1, pr_err, "iteration", "Error", 1, 1, ">", "m", "Predictive greedy", "Heat equation")
#"""

print("POD : ", Tpod, "s")
print("strong greedy : ", Tstd, "s")
print("predictive greedy : ", Tpred, "s")
#print("Reverse greedy : ", Trev, "s")

