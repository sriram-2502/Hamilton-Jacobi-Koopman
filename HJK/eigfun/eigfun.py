import numpy as np
import scipy as sp
from HJK.basis.monomials import monomials


def monomial_eigfun(x, A, U, degree):
    # To evaluate eigenfunctions at each points x in R^n
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
    # To evolve eigenfunctions at a domain X in R^nxm
    W = sp.linalg.eig(A, left=True, right=False)[1]
    Phi = W.T @ X + W.T @ (U @ Psi)

    DPhi = []
    numIC = np.shape(X)[1]
    for i in range(numIC):
        DPhi.append(W.T + W.T @ U @ DPsi[:,:,i])
    DPhi = np.reshape(DPhi, (np.shape(X)[0],-1))
    
    return [Phi, DPhi]



def monomial_eigfun_hamiltonain(x, A, U, Psi, DPsi):
    # To evolve eigenfunctions at a domain [X;P] in R^2nxm
    dim = np.shape(x)[1]

    W = sp.linalg.eig(A, left=True, right=False)[1]
    Phi = W.T @ x + W.T @ (U @ Psi)

    DPhi = []
    numIC = np.shape(x)[1]
    for i in range(numIC):
        DPhi.append(W.T + W.T @ U @ DPsi[:,:,i])
    DPhi = np.reshape(DPhi, (np.shape(X)[0],-1))

    # nonlinear x 
    PsiX = Psi[dim+1:,:]; nb_PsiX = np.size(PsiX)
    DPsiX = DPsi[dim+1:,:]

    # kronecker of linear x and linear p
    X_o_P = np.kron(Psi[:dim/2,:], Psi[dim/2:dim,:])
    nb_X_o_P = np.size(X_o_P)

    # kronecker of nonlinear x and linear p
    PsiX_o_P = np.kron(PsiX, Psi[dim/2:dim,:])
    nb_PsiX_o_P = np.size(PsiX_o_P) 

    # build basis
    Psi_Gamma = [[PsiX], [X_o_P], [PsiX_o_P]]
    nb_Psi_Gamma = np.size(Psi_Gamma)

    
    
    return [Phi, DPhi]


