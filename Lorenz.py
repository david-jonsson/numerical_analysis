import numpy as np
import matplotlib.pyplot as plt

def lorenz_system(t, y):
    """Defines the Lorenz system of ODEs."""
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dydt = np.array([
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2]
    ])
    return dydt

#------------------------------------------------------------
# Runge-Kutta 4th order method
#------------------------------------------------------------
def runge_kutta_4(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = np.zeros((1, len(y0)))
    y_values[0] = y0
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + h / 2, y0 + (h / 2) * k1)
        k3 = f(t0 + h / 2, y0 + (h / 2) * k2)
        k4 = f(t0 + h, y0 + h * k3)
        y0 += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 += h
        t_values.append(t0)
        y_values = np.vstack([y_values, y0])

    return np.array(t_values), y_values

#--------------------------------------
if __name__ == "__main__":
    # Initial conditions and parameters
    y0 = np.array([1.0, 1.0, 1.0])
    t0 = 0.0
    tf = 20.00

    t_ref, y_ref = runge_kutta_4(lorenz_system, t0, y0, t0, tf, 0.01)
    print(y_ref)

    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*y_ref.T, lw=1, color='blue')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()