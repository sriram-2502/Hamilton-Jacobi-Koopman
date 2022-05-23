import numpy as np
import control as ct
import scipy as sp
from HJK.eigfun.eigfun import monomial_eigfun

def u_HJK1(x,A,B,Q,R,U,degree):
    dim = np.shape(x)[0]
    Q = np.identity(dim)
    R = 1.0

    D, W = sp.linalg.eig(A, left=True, right=False)
    D = np.diag(D)

    R_HJK = R*(B @ B.T)
    hat_R = W.T @ R_HJK @ W
    hat_Q = np.linalg.inv(W) @ Q @ np.linalg.inv(W).T 
    L,_,_ = ct.care(D,hat_R,hat_Q)
    L_verify = W @ L @ W.T

    Phi, DPhi = monomial_eigfun(x, A, U, degree)
    

    return [L, L_verify, -(Phi.T @ L @ DPhi @ B).squeeze()]