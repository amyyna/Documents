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
from scipy.optimize import minimize


#------------------------------------------------------------------------------
"""
def perform_algorithm(X_off, N, calN):
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $, use Python to calculate the vector that satisfies 
    $$
    a = \argmin_{w \in W^+} \sum_{i=1}^N \| x_i - w \|^2,
    $$
    where $ W^+ $ is the space of vectors in $ \mathbb{R}^n $ whose elements are all non-negative.
"""

# Objective function
def objective(w, X):
    """
    Objective function to minimize.
    X: matrix of shape (N, n) where each row is a vector x_i.
    w: current vector w (needs to be a unit vector in the positive orthant).
    """
    w = w / np.linalg.norm(w)  # Ensure that w is a unit vector
    error = 0
    for x_i in X:
        projection = np.dot(x_i, w) * w  # Projection of x_i onto w
        error += np.linalg.norm(x_i - projection)**2  # Squared error
    return error

# Constraints
def constraint(w):
    """Constraint for w to be a unit vector."""
    return np.linalg.norm(w) - 1

# Non-negativity constraint
def non_negativity_constraint(w):
    """Ensures that w is in the positive orthant (non-negative)."""
    return np.min(w)

# Function to find the optimal w
def find_optimal_w(X):
    # Initial guess (starting from a non-negative random vector)
    w0 = X[0,:]
    
    # Constraints
    cons = [{'type': 'eq', 'fun': constraint},  # w must be a unit vector
            {'type': 'ineq', 'fun': non_negativity_constraint}]  # w must be non-negative
    
    # Use minimize from scipy.optimize with the objective and constraints
    result = minimize(objective, w0, args=(X,), constraints=cons, method='SLSQP')
    
    if result.success:
        return result.x / np.linalg.norm(result.x)  # Return the optimal w as a unit vector
    else:
        raise ValueError("Optimization failed")

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


w_1 = find_optimal_w(X_off.T)
print("Optimal w:", w_1)

#print("W_optimal : ", w_1)
norms = calc_norms(X_off, N)
errors = calc_error(X_off, N, w_1)
errors = calc_error(X_off, N, X_off[:,0])