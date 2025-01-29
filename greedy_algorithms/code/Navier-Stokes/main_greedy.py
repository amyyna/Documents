__Author__ = "Amina Benaceur(*)"
__Copyright__ = "September 2019 \
                   (*)MIT"

#import time
import os
import time
import numpy as np
from scipy.sparse import csc_matrix
from greedy_utils import read_vector, strong_greedy, errPlot, read_matrix, predictive_greedy, rel_POD
dir_path = "/Users/amina/Dropbox (MIT)/MIT_postdoc/greedy_theory/Navier-Stokes/"
FEM_DIR = dir_path + "Offline/fem_sols"
MAT_DIR = dir_path + "Offline/matrices"

#start = time.time()
nb_par = 50
threshold = 1e-30

#aa = os.system("freefem++ -v 0 long_river.edp")
#aa = os.system("freefem++ -v 0 fem.edp")

nb_dof, _ = read_vector(FEM_DIR+"/sol_1.txt")

KK = read_matrix(MAT_DIR+"/stiff.txt", nb_dof)

MM = read_matrix(MAT_DIR+"/mass.txt", nb_dof)

scale = 1/8.
SPM = KK + scale*MM
#SPM = np.eye(nb_dof)
#SPM = np.matrix(np.eye(min(np.shape(SPM))))
Xoff = np.matrix(np.zeros((nb_par, nb_dof)))
for jaux in range(nb_par):
    _, sol = read_vector(FEM_DIR+"/sol_"+str(jaux+1)+".txt")
    Xoff[jaux,:] = sol.T

maxModes = nb_par/4

start = time.time()
selec_params, pr_error = predictive_greedy(Xoff, SPM, threshold, nb_par, maxModes)
Tpg = time.time()-start

start = time.time()
selec_params, st_error = strong_greedy(Xoff, SPM, threshold, nb_par, maxModes)
Tsg = time.time()-start

start = time.time()
_, sg_vals, _ = rel_POD(Xoff.T, SPM, threshold, maxModes)
Tpod = time.time()-start

errPlot(np.arange(maxModes)+1, sg_vals[0:maxModes], "iteration", "Error", 1, 1, "*", "b", "POD", "Navier-Stokes")
errPlot(np.arange(maxModes)+1, st_error, "iteration", "Error", 1, 1, "*", "k", "Strong greedy", "Navier-Stokes")
errPlot(np.arange(maxModes)+1, pr_error, "iteration", "Error", 1, 1, ">", "r", "Predictive greedy", "Navier-Stokes")

print "POD : ", Tpod, "s"
print "strong greedy : ", Tsg, "s"
print "predictive greedy : ", Tpg, "s"