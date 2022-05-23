import numpy as np
import scipy as sp
from HJK.basis.monomials import monomials


def monomial_eigfun(x, A, U, degree):
    x = np.array([x]).T # add extra dimension for matmul
    dim = np.shape(x)[0]
    monomial_basis = monomials(degree)
    Psi = monomial_basis(x)
    DPsi = monomial_basis.diff(x)
    # remove 1 and linear parts
    Psi = Psi[dim+1:,:]
    DPsi = DPsi[dim+1:,:]

    W = sp.linalg.eig(A, left=True, right=False)[1]
    Phi = W.T @ x + W.T @ (U @ Psi)
    DPhi = []
    numIC = np.shape(x)[1]
    for i in range(numIC):
        DPhi.append(W.T + W.T @ U @ DPsi[:,:,i])
    DPhi = np.reshape(DPhi, (np.shape(x)[0],-1))

    return [Phi, DPhi]



def eval_monomial_eigfun(X, A, U, Psi, DPsi):
    W = sp.linalg.eig(A, left=True, right=False)[1]
    Phi = W.T @ X + W.T @ (U @ Psi)

    DPhi = []
    numIC = np.shape(X)[1]
    for i in range(numIC):
        DPhi.append(W.T + W.T @ U @ DPsi[:,:,i])
    DPhi = np.reshape(DPhi, (np.shape(X)[0],-1))
    
    return [Phi, DPhi]

