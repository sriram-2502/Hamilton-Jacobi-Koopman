import numpy as np
import control as ct

def u_LQR(x,A,B):
    dim = np.shape(x)[0]
    Q = np.identity(dim)
    R = 1.0
    K,P,_ = ct.lqr(A,B,Q,R)
    return [P, -K.dot(x).squeeze()]