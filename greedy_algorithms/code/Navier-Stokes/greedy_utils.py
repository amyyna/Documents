__Author__ = "Amina Benaceur(*)"
__Copyright__ = "September 2019 \
                   (*)MIT"
# -*- coding: utf-8 -*-  
import os
import numpy as np
from math import sqrt
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from cvxopt import solvers
from cvxopt import matrix as cvxmatrix
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def read_vector(filename):
    filin = open(filename, "r")
    lines = ''.join(filin.readlines()).strip().split()
    return int(lines[0]), np.matrix(map(float, lines[1:])).T
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def strong_greedy(Xoff, SPM, threshold, nb_modes, maxModes): 
    """ Strong greedy algorithm 
    Caution, les erreurs d'arrondi font stagner, voire legerement decoller les estimations d'erreur a partir d'un certain rang"""
    en_sqnorms = []
    for idi in range(nb_modes):
        en_sqnorms.append(Xoff[idi,:]*SPM*Xoff[idi,:].T)
    selec_params = []
    error = []
    iteration = 0
    init_ids = range(nb_modes)
    Yoff = np.matrix(Xoff)
    erreur = 10
    while (erreur>threshold and iteration<maxModes):
        error.append(sqrt(np.max(en_sqnorms)))
        idsel = np.argmax(en_sqnorms)
#        print init_ids[idsel], ", error = ", error[-1]
        selec_params.append(init_ids[idsel])
        basis_vect = Yoff[idsel,:]/sqrt(en_sqnorms[idsel])
        init_ids.pop(idsel)
        Yoff = np.delete(Yoff,idsel,axis=0)
        erreur = error[-1]
        iteration = iteration+1
        en_sqnorms = []
        for idi in range(nb_modes-iteration):
            Yoff[idi,:] = Yoff[idi,:] - float(Yoff[idi,:]*SPM*basis_vect.T)*basis_vect
            en_sqnorms.append(float(Yoff[idi,:]*SPM*Yoff[idi,:].T))
#    error.append()
    return (selec_params, error)
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def predictive_greedy(Xoff, SPM, threshold, nb_modes, maxModes):
    en_sqnorms = []
    for idi in range(nb_modes):
        en_sqnorms.append(float(Xoff[idi,:]*SPM*Xoff[idi,:].T))
    selec_params = []
    error = []
    iteration = 0
    init_ids = range(nb_modes)
    Yoff = np.matrix(Xoff)
    erreur = 10#error[0]
    while (erreur>threshold and iteration<maxModes):
        criterion = np.matrix(np.zeros((nb_modes-iteration, nb_modes-iteration)))
        maxs = []
        for mu in range(nb_modes-iteration):
            for nu in range(nb_modes-iteration):
                criterion[mu,nu] = en_sqnorms[nu] - float(Yoff[nu,:]*SPM*Yoff[mu,:].T)**2/en_sqnorms[mu]
#        for mu in range(nb_modes-iteration):
            maxs.append(np.max(criterion[mu,:]))
        error.append(sqrt(np.min(maxs)))
#        print "\n", maxs,"\n"
        idsel = np.argmin(maxs)
#        print init_ids[idsel], ", error = ", error[-1]
        basis_vect = Yoff[idsel,:]/sqrt(en_sqnorms[idsel])
        selec_params.append(init_ids[idsel])
        init_ids.pop(idsel)
        Yoff = np.delete(Yoff,idsel,axis=0)
        iteration = iteration+1
        en_sqnorms = []
        for idi in range(nb_modes-iteration):
            Yoff[idi,:] = Yoff[idi,:] - float(Yoff[idi,:]*SPM*basis_vect.T)*basis_vect
            en_sqnorms.append(float(Yoff[idi,:]*SPM*Yoff[idi,:].T))
        erreur = error[-1]
    return selec_params, error
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def errPlot(absc, vals, xl, yl, log, num, mk, cl, lab, title):
    pl.ion()
    fig = plt.figure(num)
    plt.plot(absc, vals,marker = mk, color = cl, linewidth = 2, label = lab)
    if log==1:
        plt.yscale('log')
    fig.patch.set_facecolor('white')
    plt.xticks(np.arange(1,len(vals),2))
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend()
    plt.show()
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def read_matrix(filename, nb_dof):
    filin = open(filename, "r")
    lines = ''.join(filin.readlines()[3:])
    filin.close()
    filou = open("rewritten_stiff.txt", "w")
    filou.write(lines)
    filou.close()
    matrix_file = np.matrix(np.loadtxt("rewritten_stiff.txt")).T
    matrix = csc_matrix((np.array(matrix_file[2,:]).reshape(max(np.shape(matrix_file))),
                       (np.array(matrix_file[0,:]).reshape(max(np.shape(matrix_file))),
                        np.array(matrix_file[1,:]).reshape(max(np.shape(matrix_file))))),
                      shape = (nb_dof,nb_dof))
    os.remove("rewritten_stiff.txt")
    return matrix
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def rel_POD(X_off, SPM, rel_ener, nbMod):
    """
    This function processes the POD for a given set of vectors X_off.
    It returns the first POD modes that represent an absolute energy of
    value at least equal to ener
    INPUTS      SPM   : Matrix of the considered scalar product
                nbMod : Maximum number of modes allowed
                ener  : Energy threshold
    OUTPUTS     Y     : Most significant modes
    """
    if nbMod == None:
        nbMod = min(np.shape(X_off))
    autoCorrM = X_off.T*SPM*X_off
    [eig_vals, sg_vecs]  = np.linalg.eigh(autoCorrM)
    # Sort singular values in descending order
    eig_vals = list(eig_vals)
    if abs(max(eig_vals))<1e-12:
        return (None, None, 0)
    eig_vals = [(abs(eig_value)>5e-11)*eig_value for eig_value in eig_vals]
    sg_vals = [sqrt(eig_value) for eig_value in eig_vals]
    sg_vals.reverse()
    print "------------ sings : ", sg_vals
    # Calculate the first modes (POD basis vectors)
    N0 = len(sg_vals)
    sg_vecs = np.matrix(sg_vecs)
    sum_sg = np.sum(sg_vals)
    tmp    = X_off*np.matrix(sg_vecs)[:,N0-1]
    Xpod   = tmp/sqrt(tmp.T*SPM*tmp)
    jaux   = 1
    while (jaux<=(nbMod-1) and sg_vals[jaux]/sum_sg>=rel_ener):
        tmp  = X_off*np.matrix(sg_vecs)[:,N0-1-jaux]
        # normalize  POD basis vectors
        tmp = tmp/sqrt(tmp.T*SPM*tmp)
        Xpod = np.concatenate((Xpod,tmp), axis = 1)
        jaux  = jaux+1
    trunc_order = min(jaux,nbMod)
    return (Xpod, sg_vals, trunc_order)