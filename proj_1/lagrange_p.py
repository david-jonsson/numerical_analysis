import math
import numpy as np

def get_factor(x, x_i, x_j):
    return (x - x_j) / (x_i - x_j)

def get_basis_p(x, i, points):
    basis_p = 1

    for j in range (0, len(points)):
        if (i == j): continue
        basis_p *= get_factor(x, points[i], points[j])
    
    return basis_p

def get_lagrange_p(step_size, points_x, points_y):
    """
    Computes the [len(points_x)]th order lagrange polynomial
    
    :param step_size:   step size used for sampling the lagrange polynomial
    :param points_x:    x coordinates
    :param points_y:    y coordinates
    """
    entries = int(math.fabs(points_x[-1] - points_x[0]) / step_size) + 1
    f       = np.zeros((int(entries),2))

    for n in range(0, entries):
        lagrange_p  = 0
        x           = n * step_size + points_x[0]
        for i in range(0, len(points_x)):
            lagrange_p += points_y[i] * get_basis_p(x, i, points_x)
        f[n] = (x, lagrange_p)
    
    return f

# --- Lagrange functions ---
def lagrange(x, x1, x2):
    l1 = (x2 - x) / (x2 - x1)
    l2 = (x - x1) / (x2 - x1)
    return l1, l2

# --- Piecewise linear interpolation function ---
def func_pw_linear(x, x_nodes, y_nodes):
    x = np.asarray(x)
    x_nodes = np.asarray(x_nodes)
    y_nodes = np.asarray(y_nodes)

    y = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        for j in range(len(x_nodes) - 1):
            if x_nodes[j] <= x[i] <= x_nodes[j + 1]:
                l1, l2 = lagrange(x[i], x_nodes[j], x_nodes[j + 1])
                y[i] = y_nodes[j] * l1 + y_nodes[j + 1] * l2

    return y

