import numpy as np
import Runge_Kutta as rk

angle = np.deg2rad(40)
v_0 = 40
v_x = v_0 * np.cos(angle)
v_y = v_0 * np.sin(angle)
k = 0.001
a = 0.02
g = 9.82

def y_solution(t):
    part_1 = (-v_y * k - g) * np.e ** (- k * t)
    part_2 = (-g * t + v_y) * k + g
    return  (1 / k ** 2) * (part_1 + part_2)

def x_solution(t):
    part_1 = (- v_x * k ** 2 - a - v_x) * np.e **((-k) * t) 
    part_2 = -np.cos(t) * a * k ** 2 - a * k * np.sin(t)
    part_3 = (k**2 + 1) * (a + v_x)
    fact_1 = 1 / (k * (k ** 2 + 1))
    fact_2 = part_1 + part_2 + part_3
    return fact_1 * fact_2

def f(t, u):
    """Defines the Lorenz system of ODEs."""
    dudt = np.array([
        u[1],
        -k * u[1] + a * np.sin(t),
        u[3],
        -k * u[3] - g
    ])
    return dudt

def exact_solution(t):
    y_values = []
    for i in t:
        y_values.append(np.array([x_solution(i), y_solution(i)]))
    return np.array(y_values)

y0 = np.array([0, v_x, 0, v_y])
h_values = [1, 0.5, 0.25, 0.125]

rk.convergence_study(f, y0, 0, 4, h_values, exact_solution)

