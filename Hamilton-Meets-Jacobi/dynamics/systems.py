import numpy as np

# Simple 2D system
def simple2D(x, t, u=[1.0,1.0], mu=1.0, lam=1.0):
    return [mu * x[0] + u[0], lam * (x[1] - x[0] ** 2) + u[1]]



