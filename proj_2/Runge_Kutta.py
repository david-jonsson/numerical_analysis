import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------
# Runge-Kutta 4th order method
#------------------------------------------------------------
def runge_kutta_4(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = [y0.copy()]
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + h / 2, y0 + (h / 2) * k1)
        k3 = f(t0 + h / 2, y0 + (h / 2) * k2)
        k4 = f(t0 + h, y0 + h * k3)
        y0 += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 += h
        t_values.append(t0)
        y_values.append(y0.copy())
        if y0[2] < 0.05:
            break
    
    return np.array(t_values), np.array(y_values)

#------------------------------------------------------------
# convergence study
#------------------------------------------------------------
def convergence_study(f, y00, a, b, h_values, exact_solution):
    errors_rk4 = []

    plt.figure()
    for h in h_values:
        t0 = a
        y0 = y00.copy()

        t_values, y_rk4 = runge_kutta_4(f, t0, y0, a, b, h)
        y_rk4 = np.array(y_rk4[:, [0, 2]])
                          
        y_exact = exact_solution(t_values)
        errors_rk4.append(np.max(np.abs(y_rk4 - y_exact)))
        print(f"h={h:.2e}: RK4={errors_rk4[-1]:.2e}")
        if len(errors_rk4) > 1:
            print(errors_rk4[-2], errors_rk4[-1])
            print(f"p={np.log(errors_rk4[-2] / errors_rk4[-1]) / np.log(2) }")

    plt.plot(t_values, y_rk4[:,0], color='black', linewidth=3, label='x(t)')
    plt.plot(t_values, y_rk4[:,1], color='green', linewidth=3, label='y(t)')
    plt.plot(t_values, y_exact[:,0], linewidth=3, color='yellow', label='Exact Solution x(t)', linestyle='--')
    plt.plot(t_values, y_exact[:,1], linewidth=3, color='red', label='Exact Solution y(t)', linestyle='--')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title(f'h={h}')
    plt.legend()
    plt.grid()
    plt.show()
                    
    plt.loglog(h_values, errors_rk4, marker='o', label='RK4 Error')
    plt.loglog(h_values, [h**1 for h in h_values], linestyle='--', label='O(h^1)')
    plt.loglog(h_values, [h**2 for h in h_values], linestyle='--', label='O(h^2)')
    plt.loglog(h_values, [h**3 for h in h_values], linestyle='--', label='O(h^3)')
    plt.loglog(h_values, [h**4 for h in h_values], linestyle='--', label='O(h^4)')
    plt.xlabel('Step size h')
    plt.ylabel('Max Error')
    plt.title('Convergence Study of RK Methods')
    plt.grid()
    plt.legend()
    plt.show()

def plot(f, y00, a, b, h_values):

    plt.figure()
    for h in h_values:
        t0 = a
        y0 = y00.copy()

        t_values, u_values = runge_kutta_4(f, t0, y0, a, b, h)
        plt.plot(u_values[:,0], u_values[:,2], label=h, marker='^', markersize=3)

    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.title(f'h={h}')
    plt.legend()
    plt.grid()
    plt.show()
                    
