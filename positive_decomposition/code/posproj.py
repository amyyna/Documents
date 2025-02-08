#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:18:50 2025

@author: Amina Benaceur
"""

import numpy as np
# -------------------------------- Exploratory --------------------------------
#------------------------------------------------------------------------------
def compute_largest_eigen(matrix):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Find the index of the largest eigenvalue
    largest_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    largest_eigenvalue = eigenvalues[largest_eigenvalue_index]

    # Get the associated eigenvector and normalize it
    eigenvector = eigenvectors[:, largest_eigenvalue_index]
    eigenvector_unit_norm = eigenvector / np.linalg.norm(eigenvector)
    if np.any(eigenvector_unit_norm < 0):
        eigenvector_unit_norm = -eigenvector_unit_norm

    return largest_eigenvalue, eigenvector_unit_norm

#------------------------------------------------------------------------------
def sorted_eigenvectors(matrix):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Get the indices that would sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # Sort the eigenvalues and eigenvectors accordingly
    sorted_eigenvalues = [val for val in eigenvalues[sorted_indices]]
    sorted_eigenvectors = [vec for vec in eigenvectors[:, sorted_indices]]
    for idvec, vec in enumerate(sorted_eigenvectors):   
        if np.any(vec < 0):
            #print(sorted_eigenvectors[idvec])
            sorted_eigenvectors[idvec] = -sorted_eigenvectors[idvec]
    
    return sorted_eigenvalues, sorted_eigenvectors

#------------------------------------------------------------------------------
def subtract_projection(matrix, eigenvector_unit_norm):
    # Subtract the projection of each column onto the eigenvector
    matrix_projected = matrix - np.dot(matrix, eigenvector_unit_norm[:, np.newaxis]) * eigenvector_unit_norm
    return matrix_projected


# ---------------------------------- Applied ----------------------------------
#------------------------------------------------------------------------------
def calc_pos_resid(Axi_k, col_vec_aj):
    "Calculate the positive residual vector psi^{j,k}"
    pos_proj_coef = np.dot(col_vec_aj, Axi_k)
    #print(np.shape(Axi_k), np.shape(col_vec_aj))
    adjustment_coef = np.max((pos_proj_coef*Axi_k-col_vec_aj)/Axi_k)
    xi_k_coef = (pos_proj_coef - adjustment_coef)
    psi_jk = col_vec_aj - xi_k_coef*Axi_k
    return xi_k_coef, psi_jk

#------------------------------------------------------------------------------
def assemble_pos_resid_matrix(N, AA, egvec_v_k):
    "At iteration k, assemble the matrix of the positive residual vectors psi^{j,k}, for all j"
    column_vectors = []
    egvec_coef_list = []
    Axi_k_basis = []
    for jj in range(N):
        Axi_k = np.dot(AA, egvec_v_k)
        Axi_k_basis.append(Axi_k)
        #print(np.shape(AA), np.shape(egvec_v_k), np.shape(Axi_k))
        egvec_coef, psi_jk = calc_pos_resid(Axi_k, AA[:,jj])
        column_vectors.append(psi_jk)
        egvec_coef_list.append(egvec_coef)
    new_A = np.column_stack(column_vectors)
    #print(column_vectors)
    return egvec_coef_list, new_A, Axi_k_basis
    
#------------------------------------------------------------------------------
def perform_algorithm(AA, N, nb_iterations):
    egvecs_basis = []
    AT_A = np.dot(AA.T, AA)
    compil_coef_lists = []
    for kk in range(nb_iterations):
        #print("kk = ", kk)
        lambda_1k, xi_k = compute_largest_eigen(AT_A)
        egvecs_basis.append(xi_k)
        #print(jj, xi_k)
        xi_k_coef_list, new_A, Axi_k_basis = assemble_pos_resid_matrix(N, AA, xi_k)
        compil_coef_lists.append(xi_k_coef_list)
        AT_A = np.dot(new_A.T, new_A)
        #print(np.shape(new_A), np.shape(AT_A), np.min(new_A), np.max(new_A))
    return compil_coef_lists, egvecs_basis, Axi_k_basis

#------------------------------------------------------------------------------
def reconstruct_positive_approximation(compil_coef_lists, egvecs_basis, Axi_k_basis):
    all_aj = [None]*n
    for kk in range(n):
        if kk==0:
            all_aj[kk] = [compil_coef_lists[kk][jj]*Axi_k_basis[kk] for jj in range(N)]
        else:
            all_aj = [all_aj[kk-1][jj] + compil_coef_lists[kk][jj]*Axi_k_basis[kk] for jj in range(N)]
    return all_aj
#------------------------------------------------------------------------------

# Example usage:
N = 6
calN = 100
AA = np.random.randint(1, 100, size = (calN,N))

# Step 1: Compute the largest eigenvalue and eigenvector
#first_eigenvalue, first_eigenvector = compute_largest_eigen(AA)

#eigenvalues, eigenvectors = sorted_eigenvectors(AA)

# Step 2: Subtract the projections from the matrix
#residual_matrix = subtract_projection(AA, first_eigenvector)
"""
print("Largest Eigenvalue:", first_eigenvalue)
print("Eigenvector with unit norm:\n", first_eigenvector)


print("All eigenvalues:", eigenvalues)
print("All eigenvectors\n", np.matrix(eigenvectors).T)
"""
#print("Residual matrix:\n", residual_matrix)
n = 3
compil_coef_lists, basis, Axi_k_basis = perform_algorithm(AA, N, n)
print("basis_eigen_vectors : ", basis)