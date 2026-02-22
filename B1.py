import runge_kutta
import numpy as np

h = 0.1

k = 0.001
a = 0.02
g = 9.82

angle = 40
v_0 = 40
v_x = v_0 * np.cos(angle)
v_y = v_0 * np.sin(angle)

x2_vals = None
y2_vals = None

def x1_d(t, x2):    
    return x2_vals[int(t / h)]

def x2_d(t, x2):
    return -k * x2 + a * np.sin(t)

def x_solution(t):
    part_1 = (- v_x * k ** 2 - a - v_x) * np.e **((-k) * t) 
    part_2 = -np.cos(t) * a * k ** 2 - a * k * np.sin(t)
    part_3 = (k**2 + 1) * (a + v_x)
    fact_1 = 1 / (k * (k ** 2 + 1))
    fact_2 = part_1 + part_2 + part_3
    return fact_1 * fact_2


def y1_d(t, x2):    
    return y2_vals[int(t / h)]

def y2_d(t, y2):
    return -k * y2 - g

def y_solution(t):
    part_1 = (-v_y * k - g) * np.e ** (- k * t)
    part_2 = (-g * t + v_y) * k + g
    return  (1 / k ** 2) * (part_1 + part_2)

def dummy_solution(t):
    return t * 0

t_n = np.pi * 2

dummy, x2_vals = runge_kutta.runge_kutta_4(x2_d, 0, v_x, 0, t_n, 0.1)
dummy, y2_vals = runge_kutta.runge_kutta_4(y2_d, 0, v_y, 0, t_n, 0.1)

h_values = [1, 0.5, 0.25, 0.125]



# runge_kutta.plot_solution(x1_d, 0, t_n, 0.1, x_solution)
# runge_kutta.convergence_study(x1_d, 0, t_n, h_values, x_solution)

runge_kutta.plot_solution(y1_d, 0, t_n, 1, y_solution)
runge_kutta.convergence_study(y1_d, 0, t_n, h_values, y_solution)
