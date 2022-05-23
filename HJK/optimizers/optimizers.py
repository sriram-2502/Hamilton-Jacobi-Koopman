import numpy as np
import cvxpy as cp
from casadi import *


def cvx (sys, params, A, X, Psi , DPsi):

    dim = np.shape(X)[0]
    Nbs = np.shape(Psi)[0]

    # get xdot values
    t = 0
    u = 0
    F_z = sys(0,X,0,params)

    # build gradient matrix grad_Psi * f(x)
    Df_z = []
    numIC = X.shape[1]
    for i in range(numIC):
        Df_z.append(DPsi[:,:,i] @ F_z[:,i])
    Df_z = np.array(Df_z).T

    # get linear part
    E = A
    Ez = E @ X

    # get observables and size
    G_z = Psi
    # get length of Psi to determine coefficient U dimensions
    Nbs = np.shape(G_z)[0]


    U = cp.Variable(shape=(dim, Nbs))
    objective = cp.Minimize((cp.norm(U @ Df_z + F_z - E @ U @ G_z - Ez, 'fro')))
    constraints  = []

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver='SCS', verbose=False) #conic solver


    delta = 1e-7
    nnz_l1 = (np.absolute(U.value) > delta).sum()
    print('Found a feasible x in R^{} that has {} nonzeros.'.format(dim, nnz_l1))
    print("optimal objective value: {}".format(objective.value))
    print(U.value)
    return U.value


def casadi(dim, Nbs, A, X, sys, params, Psi , DPsi):

    # get xdot values
    t = 0
    u = 0
    F_z = sys(0,X,0,params)

    # build gradient matrix grad_Psi * f(x)
    Df_z = []
    numIC = X.shape[1]
    for i in range(numIC):
        Df_z.append(DPsi[:,:,i] @ F_z[:,i])
    Df_z = np.array(Df_z).T

    # get linear part
    E = A
    Ez = E @ X

    # get observables and size
    G_z = Psi
    # get length of Psi to determine coefficient U dimensions
    Nbs = np.shape(G_z)[0]


    U = DM(dim, Nbs)
    U_sparse = U.sparse()
    f = U @ Df_z + F_z - A @ U @ G_z - Ez

    nlp = {}                # NLP declaration
    nlp['x'] = U_sparse     # decision vars
    nlp['f'] = f            # objective

    # Create solver instance
    F = nlpsol('F','ipopt',nlp);

    # Solve the problem using a guess
    F(x0=[2.5,3.0],ubg=0,lbg=0)

    return F