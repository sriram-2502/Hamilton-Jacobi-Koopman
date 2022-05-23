import numpy as np
import scipy as sp

def eigfun(X, A, U, Psi, DPsi):
    W = sp.linalg.eig(A, left=True, right=False)[1]
    Phi = W.T @ X + W.T @ (U @ Psi)

    DPhi = []
    numIC = np.shape(X)[1]
    for i in range(numIC):
        DPhi.append(W.T + W.T @ U @ DPsi[:,:,i])
    DPhi = np.reshape(DPhi, (np.shape(X)[0],-1))
    
    return [Phi, DPhi]

