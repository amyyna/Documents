__Author__ = "Amina Benaceur(*)"
__Copyright__ = "September 2019 \
                   (*)MIT"
# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from cvxopt import solvers
from cvxopt import matrix as cvxmatrix
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def read_vector(MAT_DIR, vector_name):
    filin = open(MAT_DIR+vector_name, "r")
    lines = ''.join(filin.readlines()).strip().split()
    return int(lines[0]), np.matrix(map(float, lines[1:])).T
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def read_matrix(MAT_DIR, matrix_name, nb_dof):
    filin = open(MAT_DIR+matrix_name, "r")
    lines = ''.join(filin.readlines()[3:])
    filin.close()
    filou = open(MAT_DIR+"/rewritten_stiff.txt", "w")
    filou.write(lines)
    filou.close()
    matrix_file = np.matrix(np.loadtxt(MAT_DIR+"/rewritten_stiff.txt")).T
    matrix = csc_matrix((np.array(matrix_file[2,:]).reshape(max(np.shape(matrix_file))),
                       (np.array(matrix_file[0,:]).reshape(max(np.shape(matrix_file))),
                        np.array(matrix_file[1,:]).reshape(max(np.shape(matrix_file))))),
                      shape = (nb_dof,nb_dof))
    return matrix
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def strong_greedy(Xoff, SPM, threshold, nb_modes, maxModes):
    """ Strong greedy algorithm
    Caution, les erreurs d'arrondi font stagner, voire legerement decoller les estimations d'erreur a partir d'un certain rang"""
    en_sqnorms = []
    for idi in range(nb_modes):
        en_sqnorms.append(np.dot(Xoff[idi,:], SPM.dot(Xoff[idi,:].T)))
    selec_params = []
    error = []
    iteration = 0
    init_ids = range(nb_modes)
    Yoff = np.matrix(np.copy(Xoff))
    erreur = 10
    while (erreur>threshold and iteration<maxModes):
        error.append(sqrt(max(en_sqnorms)))
        idsel = np.argmax(en_sqnorms)
        selec_params.append(init_ids[idsel])
        basis_vect = Yoff[idsel,:]/sqrt(en_sqnorms[idsel])
        init_ids.pop(idsel)
        Yoff = np.delete(Yoff,idsel,axis=0)
        erreur = error[-1]
        iteration = iteration+1
        en_sqnorms = []
        for idi in range(min(Yoff.shape)):
            #print(np.shape(Yoff[idi,:]), np.shape(SPM), np.shape(basis_vect.T))
            Yoff[idi,:] = Yoff[idi,:] - float(np.dot(Yoff[idi,:], SPM.dot(basis_vect.T)))*basis_vect
            en_sqnorms.append(float(np.dot(Yoff[idi,:], SPM.dot(Yoff[idi,:].T))))
#    error.append()
    return (selec_params, error[1:])
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def predictive_greedy(Xoff, SPM, threshold, nb_modes, maxModes):
    en_sqnorms = []
    for idi in range(nb_modes):
        en_sqnorms.append(float(np.dot(Xoff[idi,:], SPM.dot(Xoff[idi,:].T))))
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
                criterion[mu,nu] = en_sqnorms[nu] - float(np.dot(Yoff[nu,:],SPM.dot(Yoff[mu,:].T)))**2/en_sqnorms[mu]
#        for mu in range(nb_modes-iteration):
            maxs.append(np.max(criterion[mu,:]))
        error.append(sqrt(np.min(maxs)))
        #print("maxs : ", maxs)
        idsel = np.argmin(maxs)
#        print init_ids[idsel], ", error = ", error[-1]
        basis_vect = Yoff[idsel,:]/sqrt(en_sqnorms[idsel])
        selec_params.append(init_ids[idsel])
        init_ids.pop(idsel)
        Yoff = np.delete(Yoff,idsel,axis=0)
        iteration = iteration+1
        en_sqnorms = []
        for idi in range(nb_modes-iteration):
            Yoff[idi,:] = Yoff[idi,:] - float(np.dot(Yoff[idi,:], SPM.dot(basis_vect.T)))*basis_vect
            en_sqnorms.append(float(np.dot(Yoff[idi,:], SPM.dot(Yoff[idi,:].T))))
        erreur = error[-1]
    return selec_params, error
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def orthonormalize(SPM, X):
    """
    Orthonormalize the set of vectors V with respect to the matrix norm S.
    Parameters:
    S (np.ndarray): Symmetric positive-definite matrix defining the norm.
    X (np.ndarray): Matrix where each column is a vector to be orthonormalized.
    Returns:
    np.ndarray: Orthonormalized vectors, each column is a vector.
    """
    n_vectors = X.shape[0]
    orthonormal_vectors = np.matrix(np.zeros_like(X))
    for ii in range(n_vectors):
        # On commence par le vecteur actuel
        xi = X[ii, :]
        # On soustrait la projection sur les vecteurs deja orthonormalises
        for jj in range(ii):
            # Produit scalaire in the S-norm
            Xorth_j = orthonormal_vectors[jj,:]
            proj = np.dot(xi, SPM.dot(Xorth_j.T)) / np.dot(Xorth_j, SPM.dot(Xorth_j.T))
            xi = xi - proj * Xorth_j
        # On normalise par rapport a la norme definie par S
        norm = np.sqrt(float(np.dot(xi, SPM.dot(xi.T))))
        orthonormal_vectors[ii, :] = xi / norm
    return orthonormal_vectors
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def calculate_error(V, Xoff, SPM):
    """
    Calculate the error between vector V and the basis Xoff with respect to the norm defined by SPM.
    Parameters:
    V (np.ndarray): Vector to be projected, shape (n,).
    Xoff (np.ndarray): Matrix whose columns are the basis vectors, shape (n, m).
    S (np.ndarray): Positive definite matrix defining the norm, shape (n, n).
    Returns:
    float: The error (norm of the residual) in the S-norm.
    """
    # Step 1: Compute the projection of V onto the subspace spanned by Xoff
    Xoff_S_Xoff_inv = np.linalg.inv(np.dot(Xoff, SPM.dot(Xoff.T)))
    projection = np.dot(Xoff.T, np.dot(Xoff_S_Xoff_inv, np.dot(Xoff, SPM.dot(V.T))))
    # Step 2: Compute the error (residual) as the norm of V - projection
    diff = V - projection.T
    error = sqrt(np.dot(diff, SPM.dot(diff.T)))
    return error
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def get_error_decay(Xoff, selec_params, nb_params, SPM):
    X_reordered = Xoff[selec_params, :]
    X_orthonorm = orthonormalize(SPM, X_reordered)
    errors_n = []
    for ii in range(nb_params-1):
        param_errs = []
        for jj in range(ii+1, nb_params):
            err_n = calculate_error(X_reordered[jj,:], X_orthonorm[0:ii,:], SPM)
            param_errs.append(err_n)
        errors_n.append(max(param_errs))
    return errors_n
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
    autoCorrM = np.dot(X_off.T, SPM.dot(X_off))
    [eig_vals, sg_vecs]  = np.linalg.eigh(autoCorrM)
    # Sort singular values in descending order
    eig_vals = list(eig_vals)
    if abs(max(eig_vals))<1e-12:
        return (None, None, 0)
    eig_vals = [(abs(eig_value)>5e-10)*eig_value for eig_value in eig_vals]
    print("eig_vals", eig_vals)
    sg_vals = [sqrt(eig_value) for eig_value in eig_vals]
    sg_vals.reverse()
    print "------------ sings : ", sg_vals
    # Calculate the first modes (POD basis vectors)
    N0 = len(sg_vals)
    sg_vecs = np.matrix(sg_vecs)
    sum_sg = np.sum(sg_vals)
    tmp    = X_off*np.matrix(sg_vecs)[:,N0-1]
    Xpod   = tmp/sqrt(np.dot(tmp.T, SPM.dot(tmp)))
    jaux   = 1
    while (jaux<=(nbMod-1) and sg_vals[jaux]/sum_sg>=rel_ener):
        tmp  = X_off*np.matrix(sg_vecs)[:,N0-1-jaux]
        # normalize  POD basis vectors
        tmp = tmp/sqrt(np.dot(tmp.T, SPM.dot(tmp)))
        Xpod = np.concatenate((Xpod,tmp), axis = 1)
        jaux  = jaux+1
    trunc_order = min(jaux,nbMod)
    return (Xpod, sg_vals, trunc_order)
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def errPlot(absc, vals, xl, yl, log, num, mk, cl, lab, title):
    pl.ion()
    fig = plt.figure(num)
    plt.plot(absc, vals,marker = mk, color = cl, linewidth = 2, label = lab)
    if log==1:
        plt.yscale('log')
    fig.patch.set_facecolor('white')
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend()
    plt.show()
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def new_predictive_greedy(Xoff, SPM, threshold, nb_params, maxModes):
    local_Xoff = np.copy(Xoff)
    e_n = np.copy(Xoff)
    critere_to_min = []
    iteration = 0
    error = [10000]
    param_indices = range(nb_params)
    selec_params = []
    while (error[-1] > threshold  and iteration<maxModes):
        params_to_test = nb_params - iteration
        en_carre = []
        for ii in range(local_Xoff.shape[0]):
            en_carre.append(float(np.matmul(np.matmul(e_n[ii,:],SPM),e_n[ii,:].T)))
        critere_to_min = []
        for mu in range(params_to_test):
            critere_to_max = []
            for nu in range(params_to_test):
                critere_to_max.append(en_carre[nu] - (float(np.matmul(np.matmul(e_n[nu],SPM),e_n[mu].T))**2/en_carre[mu]))
            critere_to_min.append(max(critere_to_max))
        #print("mmm", critere_to_min)
        error.append(sqrt(np.min(critere_to_min)))
        current_id_optimal_parameter = np.argmin(critere_to_min)
        global_id_optimal_parameter = param_indices[current_id_optimal_parameter]
        selec_params.append(global_id_optimal_parameter)
        iteration = iteration+1
        param_indices.pop(current_id_optimal_parameter)
        np.delete(e_n, current_id_optimal_parameter, axis = 0)
        np.delete(local_Xoff, current_id_optimal_parameter, axis = 0)
        del(en_carre, critere_to_max, critere_to_min)
        basis_vector = Xoff[global_id_optimal_parameter]/sqrt(float(Xoff[global_id_optimal_parameter]*SPM*Xoff[global_id_optimal_parameter].T))
        for eta in range(nb_params-iteration):
            e_n[eta] = e_n[eta] - float(local_Xoff[eta]*SPM*Xoff[global_id_optimal_parameter].T)*basis_vector
    return selec_params, error[1:]


