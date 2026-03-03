import numpy as np
import Runge_Kutta as rk

angle = np.deg2rad(45)
v_0 = 400
v_x = v_0 * np.cos(angle)
v_y = v_0 * np.sin(angle)
k0 = 4.518e-4

def w(t):
    return (-20) * np.e ** ( (-1) * ( (t - 10) / 5) ** 2 )

def g(y):
    return (3.986 *10 ** 14) / ( (6.371 * 10 ** 6 + y) ** 2) 

def f(t, u):
    """Defines the Lorenz system of ODEs."""
    den     = np.e ** ( 1.0e-4 * u[2])
    k       = k0 / den
    common  = -k * np.sqrt( (u[1] - w(t)) ** 2 + u[3] ** 2)

    dudt = np.array([
        u[1],
        common * (u[1] - w(t)),
        u[3],
        common * u[3] - g(u[2]),
    ])
    return dudt

b = 100
y0 = np.array([0, v_x, 0, v_y])
h_values = [b / (2 ** (n)) for n in range(4, 13)]

rk.plot(f, y0, 0, b, h_values)

