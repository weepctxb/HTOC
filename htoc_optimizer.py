"""
This code is written from scratch based on the algorithm as outlined in
"Development of a Model-free Hamiltonian Tracking Optimal Control Algorithm"
by Jinkun LEE from Pennsylvania State University (2014).
This implementation allows for multivariable state vectors and scalar input.
"""

import numpy as np
import scipy.optimize as spo


# OPTIMIZER
def htoc_optimizer(x, u, t, f, lam, f_hist, lam_hist, HAM, L, dLdu, outer_iter):
    # num_dims = number of state variables / dimensions
    # num_nodes = number of time nodes / measurements
    # num_inputs = 1 = number of input variables
    # x: observed state trajectory - num_dims x num_nodes, 2D matrix
    # u: input trajectory - num_inputs (=1) x num_nodes, 1D vector
    # t: time measurements - num_nodes x 1, 1D vector
    # lam: adjoint function trajectory - num_dims x num_nodes, 2D matrix
    # f: estimated system dynamics - num_dims x num_nodes, 2D matrix
    # lam_hist: historical adjoint function trajectory (from previous outer iteration) - num_dims x num_nodes, 2D matrix
    # f_hist: historical estimated system dynamics (from previous outer iteration) - num_dims x num_nodes, 2D matrix
    # HAM: Hamiltonian trajectory - 1 x num_nodes, 1D vector
    # L: function handle describing L(x,u,t) with args x, u and t
    # dLdu: function handle describing d(L(x,u,t))/du with args x, u and t
    # outer_iter: to keep track of outer iterations. lagrangian_fast(uu) is used if outer_iter = 1
    # and lagrangian_accurate(uu) otherwise

    tol = 1e-3  # tolerance for convergence

    HAM = np.array(HAM)  # 1 x num_nodes, 1D vector
    u = np.array(u)  # num_inputs (=1) x num_nodes, 1D vector

    num_nodes = len(HAM)  # number of state variables / dimensions
    num_inputs = 1  # number of input variables

    # check if Hamiltonian time series is sufficiently constant in time
    HAM_descal = HAM / min(HAM) - 1
    HAM_lincoeff = np.polyfit(t, HAM_descal, 1)
    if np.all(np.absolute(HAM_descal) < tol) and \
        np.absolute(HAM_lincoeff[0]) < tol and np.absolute(HAM_lincoeff[1]) < tol:
        return u

    # otherwise find a new candidate input trajectory u
    else:
        HAM_target = min(HAM)

        unew = np.copy(u) # initialize u, num_inputs (=1) x num_nodes, 1D vector

        for node in range(0, num_nodes):
            # calculate dL(x,u,t)/du at time node t
            dLdu_eval = dLdu(x[:, node], u[node], t[node])  # num_inputs (=1) x 1, scalar
            yy = HAM[node] - dLdu_eval * u[node] - \
                np.dot(np.ravel(lam[:, node]), np.ravel(f[:, node]))

            # FAST: solve L(u^)=dL/du|u^+y+(Htarget-H)
            # we use this for the first outer iteration, hence the need to keep track of outer_iter
            def lagrangian_fast(uu):
                foo = - L(x[:, node], uu, t[node]) + dLdu(x[:, node], uu, t[node]) + \
                        yy + HAM_target - HAM[node]
                return foo

            # ACCURATE: solve L(u^)=dL/du|u^+y+(Htarget-H)-(lamk*fk-lam(k-1)*f(k-1))
            # we use this for subsequent outer iterations, hence the need to keep track of outer_iter
            def lagrangian_accurate(uu):
                foo = - L(x[:, node], uu, t[node]) + yy + HAM_target - HAM[node] + dLdu_eval - \
                        np.dot(np.ravel(lam[:, node]), np.ravel(f[:, node])) + \
                        np.dot(np.ravel(lam_hist[:, node]), np.ravel(f_hist[:, node]))
                return foo

            if outer_iter == 1:  # we have no historical f or lambda
                unew[node] = spo.fsolve(lagrangian_fast, u[node], xtol=1e-3)
            else:  # we now have historical f or lambda from previous outer iteration
                unew[node] = spo.fsolve(lagrangian_accurate, u[node], xtol=1e-3)

        return unew