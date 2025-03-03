#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:36:33 2025

@author: Amina Benaceur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:18:50 2025

@author: Amina Benaceur
Compute the best representatives of a cone in a hierarchical manner
"""

import cvxopt
import numpy as np


#------------------------------------------------------------------------------
def perform_algorithm(X_off, N, calN):
    """
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $, use Python to calculate the vector that satisfies 
    $$
    a = \argmin_{w \in W^+} \sum_{i=1}^N \| x_i - w \|^2,
    $$
    where $ W^+ $ is the space of vectors in $ \mathbb{R}^n $ whose elements are all non-negative.
    """
    # Formulate the quadratic objective: 
    # Sum of squared differences between x_i and w
    Q = 2 * np.eye(N) * calN  # This is the matrix for the quadratic term, since we sum squares
    c = -2 * np.sum(X_off, axis=0)  # This is the linear term
    #
    # Constraints: w >= 0 (non-negative)
    G = -np.eye(N)  # Matrix for the inequality constraint w >= 0
    h = np.zeros(N)  # Right-hand side of the inequality, 0 to enforce w >= 0
    #
    # Convert to CVXOPT format
    Q = cvxopt.matrix(Q)
    c = cvxopt.matrix(c)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    #
    # Solve the quadratic program
    sol = cvxopt.solvers.qp(Q, c, G, h)
    #
    # Extract the solution w
    w_optimal = np.array(sol['x']).flatten()
    #
    #print("Optimal w:", w_optimal)
    return w_optimal
#------------------------------------------------------------------------------
def calc_norms(X_off, N):
    """
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $ and $w_i$
    Calculate    $ \| x_i - w \|_2  $  for $i = 1, \ldots, N$
    """
    norms = []
    for ii in range(N):
        norms.append(np.linalg.norm(X_off[:,ii]))
    print("max norm = ", np.max(norms))
    return norms
#------------------------------------------------------------------------------
def calc_error(X_off, N, w_i):
    """
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $ and $w_i$
    Calculate    $ \| x_i - w \|_2  $  for $i = 1, \ldots, N$
    """
    errors = []
    for ii in range(N):
        errors.append(np.linalg.norm(X_off[:,ii] - w_i))
    print("max error = ", np.max(errors))
    return errors

#------------------------------------------------------------------------------

# Example 
N = 6  # Number of vectors
calN = 100  # Dimension of each vector
X_off = np.random.randn(calN,N) # Randomly generated X_off

w_1 = perform_algorithm(X_off.T, calN, N)
#print("W_optimal : ", w_1)
norms = calc_norms(X_off, N)
errors = calc_error(X_off, N, w_1)
errors = calc_error(X_off, N, X_off[:,0])