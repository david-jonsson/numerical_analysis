import numpy as np
import Runge_Kutta as rk
import matplotlib.pyplot as plt

v_0 = 400
k0 = 4.518e-4

def w(t):
    return (-20) * np.e ** ( (-1) * ( (t - 10) / 5) ** 2 )

def g(y):
    return (3.986 *10 ** 14) / ( (6.371 * 10 ** 6 + y) ** 2) 

def u(t, u):
    """Defines the Lorenz system of ODEs."""
    den     = np.e ** ( 1.0e-4 * u[2])
    k       = k0 / den
    common  = -k * np.sqrt( (u[2] - w(t)) ** 2 + u[3] ** 2)

    dudt = np.array([
        u[1],
        common * (u[1] - w(t)),
        u[3],
        common * u[3] - g(u[2]),
    ])
    return dudt

def f(angle):
    v_x = v_0 * np.cos(angle)
    v_y = v_0 * np.sin(angle)

    y0 = np.array([0, v_x, 0, v_y])
    t_values, u_values = rk.runge_kutta_4(u, 0, y0, 0, b, 1.5)

                    
    plt.plot(u_values[:,0], u_values[:,2], label=np.mod(np.degrees(angle), 360), marker='^', markersize=3)
    return np.abs(u_values[-1][0] - 2700)


def f_prim(angle, delta):
    return (f(angle + delta) - f(angle - delta)) / (2 * delta)


b = 100
angle = np.deg2rad(45)
delta = 1e-6

plt.figure()
while 1:
    fn = f(angle)
    # print(fn, angle)

    if fn < 10:
        print(np.mod(np.degrees(angle), 360))
        break; 
    else:
        angle = angle - fn / f_prim(angle, delta)


plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

v_x = v_0 * np.cos(angle)
v_y = v_0 * np.sin(angle)
y0 = np.array([0, v_x, 0, v_y])
h_values = [1.5]

