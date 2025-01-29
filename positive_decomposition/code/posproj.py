#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:18:50 2025

@author: Amina Benaceur
"""

import numpy as np

def compute_largest_eigen(matrix):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Find the index of the largest eigenvalue
    largest_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    largest_eigenvalue = eigenvalues[largest_eigenvalue_index]

    # Get the associated eigenvector and normalize it
    eigenvector = eigenvectors[:, largest_eigenvalue_index]
    eigenvector_unit_norm = eigenvector / np.linalg.norm(eigenvector)
    if np.any(eigenvector_unit_norm < 0):
        eigenvector_unit_norm = -eigenvector_unit_norm

    return int(largest_eigenvalue), np.real(eigenvector_unit_norm)

def subtract_projection(matrix, eigenvector_unit_norm):
    # Subtract the projection of each column onto the eigenvector
    matrix_projected = matrix - np.dot(matrix, eigenvector_unit_norm[:, np.newaxis]) * eigenvector_unit_norm
    return matrix_projected

def sorted_eigenvectors(matrix):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Get the indices that would sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # Sort the eigenvalues and eigenvectors accordingly
    sorted_eigenvalues = [int(val) for val in eigenvalues[sorted_indices]]
    sorted_eigenvectors = [np.real(vec) for vec in eigenvectors[:, sorted_indices]]
    for idvec, vec in enumerate(sorted_eigenvectors):   
        if np.any(vec < 0):
            print(sorted_eigenvectors[idvec])
            sorted_eigenvectors[idvec] = -sorted_eigenvectors[idvec]
    
    return sorted_eigenvalues, sorted_eigenvectors

# Example usage:
n = 4
matrix = np.random.randint(1, 100, size = (n,n))

# Step 1: Compute the largest eigenvalue and eigenvector
first_eigenvalue, first_eigenvector = compute_largest_eigen(matrix)

eigenvalues, eigenvectors = sorted_eigenvectors(matrix)

# Step 2: Subtract the projections from the matrix
residual_matrix = subtract_projection(matrix, first_eigenvector)

print("Largest Eigenvalue:", first_eigenvalue)
print("Eigenvector with unit norm:\n", first_eigenvector)


print("All eigenvalues:", eigenvalues)
print("All eigenvectors\n", np.matrix(eigenvectors).T)

#print("Residual matrix:\n", residual_matrix)