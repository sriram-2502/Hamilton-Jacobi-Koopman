import numpy as np


def simple2D(x, t, u=[1.0, 1.0], mu=1.0, lam=1.0):
    x1 = x[0]
    x2 = x[1]

    B = [1, 1]
    u1 = B[0]*u[0]
    u2 = B[1]*u[1]

    return np.array([mu*x1 + u1, lam*(x2-x1**2) + u2])


def simple2Dsystem(t, x, u, params):
    x1 = x[0]
    x2 = x[1]

    B = [1.0, 1.0]
    u1 = B[0]*u
    u2 = B[1]*u

    mu = params.get('mu',-1.0)
    lam = params.get('lam',-2.0)

    dx1 = mu * x1 + u1
    dx2 = lam * (x2 - x1 ** 2) + u2
    return np.array([dx1 , dx2])




def oscillator2D(x, t, u=1.0):
    return np.array([x[2], -x[0] + 0.105*x[1] + 0.5*x[1]*x[0]**2 + 1.1*x[0]*x[1] + 1.1+x[0]*u])

def mems3D(x, t, u=1.0):
    return np.array([x[1], -x[0]+x[1]-x[2]-(x[0]**2)*x[1], x[2]-x[2]-x[2]**3])



# Runge Kutta first order (Euler) approximation of the ode solution 
def rk1(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i+1] = y[i] + (t[i+1] - t[i]) * f(y[i], t[i], *args)
    return y



# Runge Kutta fourth order approximation of the ode solution
def rk4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y