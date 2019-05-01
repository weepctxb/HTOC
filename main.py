"""
This code is written from scratch based on the algorithm as outlined in
"Development of a Model-free Hamiltonian Tracking Optimal Control Algorithm"
by Jinkun LEE from Pennsylvania State University (2014).
This implementation allows for multivariable state vectors and scalar input.
"""

from htoc_partsysident import htoc_partsysident
from htoc_optimizer import htoc_optimizer
import numpy as np

# RETRIEVE x, xh, u, t HERE
# TODO: Specify an initial guess of input trajectory u
u = [4, 6, 8, 10, 12, 14]  # igit config -- global user.name EPCTechnologiesnput trajectory - num_inputs (=1) x num_nodes, 1D vector
def retrieveobservation(u):
    # TODO: Complete this to retrieve state and perturbed state trajectories, x and xh, from the RNN
    x = [[1., 2., 3., 4., 5., 6.], [4., 5., 6., 7., 8., 9.]]
    xh = [[[1.14, 2.15, 3.35, 4.44, 5.49, 6.57], [4.18, 5.31, 6.62, 7.77, 8.98, 10.01]], # perturb in x1
          [[1.23, 2.07, 3.31, 4.39, 5.46, 6.58], [4.21, 5.28, 6.67, 7.87, 9.03, 10.14]]] # perturb in x2
    # convert to numpy matrices first
    x = np.matrix(x)  # observed state trajectory - num_dims x num_nodes, 2D matrix
    xh = np.stack(xh)  # perturbed state trajectory - num_pert x num_dims x num_nodes, 3D matrix
    return x, xh

t = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7]  # time measurements - num_nodes x 1, 1D vector

# TERMINAL COST STATE DERIVATIVE
def dphidx(x, t):
    # TODO: Define terminal cost function derivative here
    return 1.  # return as scalar

# RECURRING COST STATE DERIVATIVE
def dLdx(x, u, t):
    num_dims = len(x)
    # TODO: Define recurring cost function state derivative here
    return np.zeros(num_dims)  # return as num_dims x 1, 1D vector

# RECURRING COST INPUT DERIVATIVE
def dLdu(x, u, t):
    # TODO: Define recurring cost function input derivative here
    return 0.  # return as scalar

# RECURRING COST
def L(x, u, t):
    # TODO: Define recurring cost function here
    return 0.5  # return as scalar

# INITIALIZE X, XH AND HAM
x, xh = retrieveobservation(u)
HAM = np.arange(len(t))+1.  # just a dummy non-constant non-zero vector

# INITIALIZE HISTORICAL VARIABLES AND FLAG
f_hist = np.zeros(np.shape(x))
lam_hist = np.zeros(np.shape(x))
HAM_hist = 2.*np.arange(len(t))+1.  # just a dummy non-constant non-zero vector sufficiently different from HAM
iter1 = True

# OUTER ITERATION - while H[time] not constant with time
HAM_descal = HAM / min(HAM) - 1
HAM_lincoeff = np.polyfit(t, HAM_descal, 1)
tol = 1e-3
outer_iter = 1
while (np.any(np.absolute(HAM_descal) > tol) or
       np.absolute(HAM_lincoeff[0]) > tol or np.absolute(HAM_lincoeff[1]) > tol) and outer_iter < 1e3:
    x, xh = retrieveobservation(u)
    f, lam, HAM = htoc_partsysident(x, xh, u, t, dphidx, dLdx, L)

    # INNER ITERATION - while H[time] not converged
    inner_iter = 1
    while any(np.absolute((HAM - HAM_hist) / HAM_hist) > tol) and inner_iter < 1e3:
        u = htoc_optimizer(x, u, t, f, lam, f_hist, lam_hist, HAM, L, dLdu, outer_iter)
        HAM_hist = np.copy(HAM)
        _, _, HAM = htoc_partsysident(x, xh, u, t, dphidx, dLdx, L)
        inner_iter += 1
    f_hist = np.copy(f)
    lam_hist = np.copy(lam)
    outer_iter += 1

# REPORTING
print("f = ",f)
print("lambda = ",lam)
print("HAM = ",HAM)
print("u = ",u)
print("Outer iterations = ",outer_iter)
print("Inner iteration last count = ",inner_iter)
