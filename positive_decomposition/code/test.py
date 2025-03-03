import numpy as np
from scipy.optimize import minimize

#------------------------------------------------------------------------------
def calc_norms(X_off, N):
    """
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $ and $w_i$
    Calculate    $ \| x_i \|_2  $  for $i = 1, \ldots, N$
    """
    norms = []
    for ii in range(N):
        norms.append(np.linalg.norm(X_off[ii,:]))
    print("max norm = ", np.max(norms))
    return norms

#------------------------------------------------------------------------------
def calc_rel_error(X_off, N, w_i):
    """
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $ and $w_i$
    Calculate  the relative error  $ \| x_i - np.dot(x_i, w)*w \|_2 / \|x_i\|_2  $  for $i = 1, \ldots, N$
    """
    errors = []
    for jj in range(N):
        relative_error = np.linalg.norm(X_off[jj,:] - np.dot(X_off[jj,:], w_i)*w_i)/np.linalg.norm(X_off[jj,:]) # This is not optimal, you can collect the previously computed norms
        errors.append(relative_error)
    print("max error = ", np.max(errors))
    return errors

#------------------------------------------------------------------------------
def calc_abs_error(X_off, N, w_i):
    """
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $ and $w_i$
    Calculate the absolute error  $ \| x_i - np.dot(x_i, w)*w \|_2 $  for $i = 1, \ldots, N$
    """
    errors = []
    for jj in range(N):
        relative_error = np.linalg.norm(X_off[jj,:] - np.dot(X_off[jj,:], w_i)*w_i) # This is not optimal, you can collect the previously computed norms
        errors.append(relative_error)
    print("max error = ", np.max(errors))
    return errors

#------------------------------------------------------------------------------
def exec_iteration(X_prev, N, w_i):
    """
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $ and a list of vectors $w_1,...,w_i$
    Calculate the next vector in the approximation space
    """
    X_resid = np.matrix(np.zeros_like(X_prev))
    for jj in range(N):
        X_resid[jj,:] = X_prev[jj,:] - np.max(0, np.dot(X_prev[jj,:], w_i))*w_i# This is not optimal, this residual has been computed elsewhere
    #Initialization
    w0 = X_resid[0,:]  # Starting with a vector of ones, but we will project later  
    # Constraints for optimization
    constraints = [
        {'type': 'ineq', 'fun': constraint_non_neg},  # Non-negative constraint
        {'type': 'eq', 'fun': constraint_unit_norm},  # Unit norm constraint
    ]
    # Optimize the objective function
    result = minimize(objective, w0, args=(X_resid,), constraints=constraints, method="trust-constr")# "Nelder-Mead")#'SLSQP'#trust-constr is the only one that satisfies norm=1, SLSQ follows with norm = 1.4, and the rest do not satisfy it
    return result


#------------------------------------------------------------------------------
def perform_iteration(X_prev, N, w_i):
    """
    Given a set of vectors $ (x_1, \dots, x_N) $ in $ \mathbb{R}^n $ and a list of vectors $w_1,...,w_i$
    Calculate the next vector in the approximation space
    """
    
    for jj in range(N):
        
    X_resid = np.matrix(np.zeros_like(X_prev))
    for jj in range(N):
        X_resid[jj,:] = X_prev[jj,:] - np.max(0, np.dot(X_prev[jj,:], w_i))*w_i# This is not optimal, this residual has been computed elsewhere
    #Initialization
    w0 = X_resid[0,:]  # Starting with a vector of ones, but we will project later  
    # Constraints for optimization
    constraints = [
        {'type': 'ineq', 'fun': constraint_non_neg},  # Non-negative constraint
        {'type': 'eq', 'fun': constraint_unit_norm},  # Unit norm constraint
    ]
    # Optimize the objective function
    result = minimize(objective, w0, args=(X_resid,), constraints=constraints, method="trust-constr")# "Nelder-Mead")#'SLSQP'#trust-constr is the only one that satisfies norm=1, SLSQ follows with norm = 1.4, and the rest do not satisfy it
    return result

#------------------------------------------------------------------------------
# Define the objective function
def objective(w, X):
    """
    Objective function to minimize.
    w: A vector in R^n (needs to be non-negative and of unit norm)
    X: A matrix of size N x n where each row is a vector x_i.
    """
    w = w / np.linalg.norm(w)  # Ensure w is a unit vector
    residual = 0
    for x in X:
        #print(np.shape(x))
        proj = np.dot(x, w) * w
        residual += np.linalg.norm(x - proj)**2
    return residual

#------------------------------------------------------------------------------
# Define the constraints: w should be non-negative and have unit norm
def constraint_non_neg(w):
    return w

#------------------------------------------------------------------------------
def constraint_unit_norm(w):
    return np.linalg.norm(w) - 1  # This will ensure the unit norm constraint

#------------------------------------------------------------------------------
# Set the random seed for reproducibility
np.random.seed(2) 

# Example data: N vectors of dimension n
calN = 200  # Number of vectors
N = 7  # Dimensionality of each vector
X = np.random.rand(N, calN)  # Randomly generated vectors in R^n

# Initial guess for w (random vector)
w0 = X[0,:]  # Starting with a vector of ones, but we will project later

# Constraints for optimization
constraints = [
    {'type': 'ineq', 'fun': constraint_non_neg},  # Non-negative constraint
    {'type': 'eq', 'fun': constraint_unit_norm},  # Unit norm constraint
]

# Optimize the objective function
result = minimize(objective, w0, args=(X,), constraints=constraints, method="trust-constr")# "Nelder-Mead")#'SLSQP'#trust-constr is the only one that satisfies norm=1, SLSQ follows with norm = 1.4, and the rest do not satisfy it

print("norm = ", np.linalg.norm(result.x))
print("positivity min check : ", np.min(result.x))
# The optimized w
w_1 = result.x / np.linalg.norm(result.x)  # Ensure it remains a unit vector

#print("Optimized vector w:", w_1)

norms = calc_norms(X, N)
errors_w1 = calc_rel_error(X, N, w_1)
errors_X1 = calc_rel_error(X, N, X[0,:])
print("norms : ", norms)
print("\nerrors w1 : ", errors_w1)
print("\nerrors X1 : ", errors_X1)

