__Author__ = "Amina Benaceur(*)"
__Copyright__ = "December 2024' \
                   (*)UM6P"
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sc
from math import sqrt
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
#from cvxopt import solvers
#from cvxopt import matrix as cvxmatrix
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def read_vector(MAT_DIR, vector_name):
    filin = open(MAT_DIR+vector_name, "r")
    lines = ''.join(filin.readlines()).strip().split()
    #print(lines[1:])
    return int(lines[0]), np.matrix([float(ii) for ii in lines[1:]]).T
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
def orthogonal_projection(vec, X_out, SPM):
    """
    Perform the orthogonal projection of vector `vec` onto the subspace spanned by columns of `X_out`,
    with respect to the scalar product matrix `SPM`.
    Parameters:
    v (numpy array): The vector to be projected.
    X_out (numpy array): A matrix whose columns are the vectors spanning the subspace.
    SPM (numpy array): The scalar product matrix.
    Returns:
    numpy array: The orthogonal projection of `vec` onto the subspace.
    """
    inv_SPM = sc.sparse.linalg.inv(SPM)
    # Compute the projection using the formula: p = X_out * (X_out.T * SPM * X_out)^(-1) * X_out.T * SPM * v
    #print("honaa", np.shape(X_out), np.shape(X_out.T), np.shape(SPM) ,np.shape(X_out), np.shape(X_out.T), np.shape(inv_SPM), np.shape(vec))
    projection = np.dot(X_out, np.dot(np.linalg.inv(np.dot(X_out.T, inv_SPM.dot(X_out))), np.dot(X_out.T, inv_SPM.dot(vec))))
    return projection
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def reverse_greedy(Xoff, SPM, threshold, nb_modes, maxModes):
    """ Reverse greedy algorithm maximizing projections over the leftover vectors (X_out)"""
    all_params = list(range(nb_modes))
    in_params = []
    out_params = all_params.copy()
    X_out = np.matrix(np.copy(Xoff)) #all prams are out at initial stage
    X_in = None
    error = []
    flag = True
    iteration = 0
    while flag:
        proj_errors = [] #projection error on the urrent X_n
        if iteration == 0:
            for idi_out, param_out in enumerate(out_params):
                #print("aqui ", np.shape(X_out[:,idi_out]))
                proj_errors.append(sqrt(np.dot(X_out[:,idi_out].T, SPM.dot(X_out[:,idi_out]))))
        else:
            for idi_out, param_out in enumerate(out_params):
                proj_errors.append(calculate_error(X_out[:, idi_out].T, X_in.T, SPM))
        error.append(np.max(proj_errors))
        proj_criterion = []
        for idi in range(min(np.shape(X_out))):
            if idi == 0:
                X_out_next = X_out[:, idi+1:]
            else:
                #print("here ", np.shape(X_out[:, :idi]), np.shape(X_out[:, idi+1:]))
                X_out_next = np.concatenate((X_out[:, :idi], X_out[:, idi+1:]), axis = 1)
            #print(np.shape(X_out[:,idi]), np.shape(X_out_next))
            #print(idi, " -- ", iteration)
            proj_on_Xout_next = orthogonal_projection(X_out[:,idi], X_out_next, SPM)
            #print(np.shape(proj_on_Xout_next))
            proj_criterion.append(np.dot(proj_on_Xout_next.T, SPM.dot(proj_on_Xout_next)))
        local_idsel = np.argmax(proj_criterion)
        global_idsel = out_params[local_idsel]
        in_params.append(global_idsel)
        basis_vect = X_out[:, local_idsel]/sqrt(float(np.dot(X_out[:, local_idsel].T, SPM.dot(X_out[:, local_idsel]))))
        out_params.pop(local_idsel)
        X_out = np.delete(X_out, local_idsel, axis=1)
        if iteration==0:
            X_in = basis_vect
        else:
            X_in = np.concatenate((X_in, basis_vect), axis = 1)
        iteration = iteration + 1
        erreur = error[-1]
        for ind in range(min(X_out.shape)):
            #print(np.shape(X_out[:, ind]), np.shape(SPM), np.shape(basis_vect))
            X_out[:, ind] = X_out[:, ind] - float(np.dot(X_out[:, ind].T, SPM.dot(basis_vect)))/sqrt(float(np.dot(X_out[:, ind].T, SPM.dot(X_out[:, ind]))))*basis_vect
        if (erreur < threshold or iteration >= maxModes):
            flag = False
    return (in_params, error)
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def standard_greedy(Xoff, SPM, threshold, nb_modes, maxModes):
    """ Standard greedy algorithm
    Caution, les erreurs d'arrondi font stagner, voire legerement decoller les estimations d'erreur a partir d'un certain rang"""
    en_norms = []
    for kk in range(nb_modes):
        en_norms.append(sqrt(float(np.dot(Xoff[kk,:], SPM.dot(Xoff[kk,:].T)))))
    selec_params = []
    error = []
    iteration = 0
    init_ids = list(range(nb_modes))
    Yoff = np.matrix(np.copy(Xoff))
    erreur = 10#any large value works. We only use it to enter the loop
    while (erreur>threshold and iteration<maxModes):
        error.append(max(en_norms))
        idsel = np.argmax(en_norms)
        selec_params.append(init_ids[idsel])
        basis_vect = Yoff[idsel,:]/sqrt(float(np.dot(Yoff[idsel,:], SPM.dot(Yoff[idsel,:].T))))
        init_ids.pop(idsel)
        Yoff = np.delete(Yoff,idsel, axis=0)
        erreur = error[-1]
        iteration = iteration+1
        en_norms = []
        for idi in range(min(Yoff.shape)):
            Yoff[idi,:] = Yoff[idi,:] - float(np.dot(Yoff[idi,:], SPM.dot(basis_vect.T)))*basis_vect
            en_norms.append(sqrt(float(np.dot(Yoff[idi,:], SPM.dot(Yoff[idi,:].T)))))
    return (selec_params, error)
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
def predictive_greedy(Xoff, SPM, threshold, nb_modes, maxModes):
    en_squared_norms = []
    for kk in range(nb_modes):
        en_squared_norms.append(float(np.dot(Xoff[kk,:], SPM.dot(Xoff[kk,:].T))))
    selec_params = []
    error = []
    iteration = 0
    init_ids = list(range(nb_modes))
    Yoff = np.matrix(np.copy(Xoff))
    erreur = sqrt(np.max(en_squared_norms))
    error.append(erreur)
    while (erreur>threshold and iteration<maxModes):
        criterion = np.matrix(np.zeros((nb_modes-iteration, nb_modes-iteration)))
        maxs = []
        for mu in range(nb_modes-iteration):
            for nu in range(nb_modes-iteration):
                criterion[mu,nu] = en_squared_norms[nu] - float(np.dot(Yoff[nu,:],SPM.dot(Yoff[mu,:].T)))**2/en_squared_norms[mu]
#        for mu in range(nb_modes-iteration):
            maxs.append(np.max(criterion[mu,:]))
        #print("maxs", maxs)
        error.append(sqrt(np.min(maxs)))
        idsel = np.argmin(maxs)
        selec_params.append(init_ids[idsel])
        basis_vect = Yoff[idsel,:]/sqrt(float(np.dot(Yoff[idsel,:], SPM.dot(Yoff[idsel,:].T))))
        init_ids.pop(idsel)
        Yoff = np.delete(Yoff,idsel,axis=0)
        erreur = error[-1]
        iteration = iteration+1
        en_squared_norms = []
        for idi in range(nb_modes-iteration):
            Yoff[idi,:] = Yoff[idi,:] - float(np.dot(Yoff[idi,:], SPM.dot(basis_vect.T)))*basis_vect
            en_squared_norms.append(float(np.dot(Yoff[idi,:], SPM.dot(Yoff[idi,:].T))))
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
    #print(np.shape(Xoff), np.shape(SPM), np.shape(Xoff.T))
    Xoff_S_Xoff_inv = np.linalg.inv(np.dot(Xoff, SPM.dot(Xoff.T)))
    #print(np.shape(Xoff.T), np.shape(Xoff_S_Xoff_inv), np.shape(Xoff), np.shape(SPM), np.shape(V.T))
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
    print("------------ sings : ", sg_vals)
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


