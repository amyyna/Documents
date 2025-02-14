#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:46:38 2025

@author: benaceur
"""
import cvxopt
import numpy as np

# Example data: N points, each in M-dimensional space
N = 10  # Number of points (a_j)
M = 5   # Dimension of each x and a_j
np.random.seed(42)

# Random data for a_j (N x M matrix)
a = np.random.rand(N, M)

a = np.matrix(np.eye((2)))

# Convert data to CVXOPT matrices
a = np.array(a)
a_sum = np.sum(a, axis=0)

# Define the quadratic part of the objective (P = 2N * I)
P = 2 * N * np.eye(M)
P = cvxopt.matrix(P)

# Define the linear part of the objective (q = -2 * sum(a_j))
q = -2 * a_sum
q = cvxopt.matrix(q)

# Define the constraints (if any)
# No explicit constraints in this case, so G = 0 and h = 0

# Create and solve the QP problem
problem = cvxopt.qp(P, q)

# Extract the optimal solution (x)
x_opt = np.array(problem['x']).flatten()

# Print the result
print("Optimal x:", x_opt)
