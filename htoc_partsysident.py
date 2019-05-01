"""
This code is written from scratch based on the algorithm as outlined in
"Development of a Model-free Hamiltonian Tracking Optimal Control Algorithm"
by Jinkun LEE from Pennsylvania State University (2014).
This implementation allows for multivariable state vectors and scalar input.
April 2019
"""

import numpy as np


# PARTIAL SYSTEM IDENTIFIER
def htoc_partsysident(x, xh, u, t, dphidx, dLdx, L):
    # num_dims = number of state variables / dimensions
    # num_perts = num_dims = number of possible perturbations
    # num_nodes = number of time nodes / measurements
    # num_inputs = 1 = number of input variables
    # x: observed state trajectory - num_dims x num_nodes, 2D matrix
    # xh: perturbed state trajectory - num_pert x num_dims x num_nodes, 3D matrix
    # u: input trajectory - num_inputs (=1) x num_nodes, 1D vector
    # t: time measurements - num_nodes x 1, 1D vector
    # dphidx: function handle describing d(phi(x,t))/dx with args x and t
    # dLdx: function handle describing d(L(x,u,t))/dx with args x, u and t
    # L: function handle describing L(x,u,t) with args x, u and t

    x = np.matrix(x)  # num_dims x num_nodes, 2D matrix
    xh = np.stack(xh)  # num_pert x num_dims x num_nodes, 3D matrix
    u = np.array(u)  # num_inputs (=1) x num_nodes, 1D vector

    # throw exception if x, u and t have different number of nodes
    if not((x.shape[1] == len(u))
           and (x.shape[1] == len(t))  # x and u have same number of time nodes
           and (x.shape[0] == xh.shape[1])  # x and x^ have same number of dimensions
           and (xh.shape[0] == xh.shape[1])):  # number of perturbations and dimensions are the same
        raise Exception('Dimensions of x, x^, u and t do not match!')

    num_dims = x.shape[0]  # number of state variables / dimensions
    num_perts = num_dims  # = number of possible perturbations
    num_inputs = 1  # number of input variables
    num_nodes = len(t)  # number of time nodes

    dt = np.diff(t)  # generate the 1st order difference in time

    f = np.zeros((num_dims, num_nodes))  # initialize f(x,u,t), num_dims x num_nodes
    fh = np.zeros((num_dims, num_dims, num_nodes))  # initialize f(x^,u,t), num_dims x num_dims x num_nodes

    # Estimate dynamics f(x,u,t)
    for dim in range(0, num_dims):
        for node in range(0, num_nodes):
            if node == 0:
                f[dim, node] = (x[dim, 1] - x[dim, 0]) / dt[0]
            elif node == num_nodes - 1:
                f[dim, node] = (x[dim, node] - x[dim, node - 1]) / dt[node - 1]
            else:
                f[dim, node] = (x[dim, node + 1] - x[dim, node - 1]) / (dt[node] + dt[node - 1])

    # Estimate perturbed dynamics f(x^,u,t)
    for pert in range(0, num_perts):
        for dim in range(0, num_dims):
            for node in range(0, num_nodes):
                if node == 0:
                    fh[pert, dim, node] = (xh[pert, dim, 1] - xh[pert, dim, 0]) / dt[0]
                elif node == num_nodes - 1:
                    fh[pert, dim, node] = (xh[pert, dim, node] - xh[pert, dim, node - 1]) / dt[node - 1]
                else:
                    fh[pert, dim, node] = (xh[pert, dim, node + 1] - xh[pert, dim, node - 1]) / (dt[node] + dt[node - 1])

    # Estimate perturbation Jacobian df/dx
    dfdx = np.zeros((num_nodes, num_dims, num_perts))  # initialize dfdx, num_nodes x num_dims x num_perts

    for node in range(0, num_nodes):
        for dim in range(0, num_dims):
            for pert in range(0, num_perts):
                dfdx[node, dim, pert] = (fh[pert, dim, node] - f[dim, node]) / (xh[pert, dim, node] - x[dim, node])

    # Estimate adjoint function lam
    lam = np.zeros((num_dims, num_nodes))  # initialize lam, num_dims x num_nodes, 2D matrix
    for node in range(num_nodes - 1, -1, -1):  # calculate adjoint function backwards in time
        if node == num_nodes - 1:
            # boundary condition for adjoint function
            lam[:, node] = dphidx(np.ravel(x[:, node]), t[node])
        else:
            # calculate dL(x,u,t)/dx at time node t
            dLdx_eval = dLdx(x[:, node + 1], u[node + 1], t[node + 1])  # num_dims x 1, 1D vector
            # calculate adjoint function backwards in time
            for dim in range(0, num_dims):
                lam[dim, node] = lam[dim, node + 1] + \
                                 dt[node] * np.dot(np.ravel(dfdx[node + 1, dim, :]), np.ravel(lam[:, node + 1])) + \
                                 dt[node] * dLdx_eval[dim]

    # Finally determine Hamiltonian HAM
    HAM = np.zeros(num_nodes)  # initialize HAM, 1 x num_nodes, 1D vector
    for node in range(0, num_nodes):
        HAM[node] = np.dot(np.ravel(lam[:, node]), np.ravel(f[:, node])) + L(x[:,node], u[node], t[node])

    return f, lam, HAM