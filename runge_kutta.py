import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------
# Runge-Kutta 1nd order method
#------------------------------------------------------------
def runge_kutta_1(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = [y0]
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        y0 += h * k1
        t0 += h
        t_values.append(t0)
        y_values.append(y0)

    return np.array(t_values), np.array(y_values)


#------------------------------------------------------------
# Runge-Kutta 2nd order method
#------------------------------------------------------------
def runge_kutta_2(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = [y0]
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + h, y0 + h * k1)
        y0 += (h / 2) * (k1 + k2)
        t0 += h
        t_values.append(t0)
        y_values.append(y0)

    return np.array(t_values), np.array(y_values)


#------------------------------------------------------------
# Runge-Kutta 3rd order method
#------------------------------------------------------------
def runge_kutta_3(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = [y0]
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + h / 2, y0 + (h / 2) * k1)
        k3 = f(t0 + h, y0 + h * k2)
        y0 += (h / 6) * (k1 + 4 * k2 + k3)
        t0 += h
        t_values.append(t0)
        y_values.append(y0)

    return np.array(t_values), np.array(y_values)   

#--------------------------------------
# Runge-Kutta 3rd order method
#--------------------------------------
def runge_kutta_33(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = [y0]
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + h / 3, y0 + (h / 3) * k1)
        k3 = f(t0 + 2/3*h, y0 + 2/3*h * k2)
        y0 += h *(1/4 * k1 + 3/4 * k3)
        t0 += h
        t_values.append(t0)
        y_values.append(y0)

    return np.array(t_values), np.array(y_values)


#------------------------------------------------------------
# Runge-Kutta 4th order method
#------------------------------------------------------------
def runge_kutta_4(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = [y0]
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + h / 2, y0 + (h / 2) * k1)
        k3 = f(t0 + h / 2, y0 + (h / 2) * k2)
        k4 = f(t0 + h, y0 + h * k3)
        y0 += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 += h
        t_values.append(t0)
        y_values.append(y0)

    return np.array(t_values), np.array(y_values)

#------------------------------------------------------------
# convergence study
#------------------------------------------------------------
def convergence_study(f, a, b, h_values, exact_solution):
    t0 = a
    y0 = 0

    errors_rk1 = []
    errors_rk2 = []
    errors_rk3 = []
    errors_rk4 = []

    plt.figure()
    for h in h_values:
        t_values, y_rk1 = runge_kutta_1(f, t0, y0, a, b, h)
        t_values, y_rk2 = runge_kutta_2(f, t0, y0, a, b, h)
        t_values, y_rk3 = runge_kutta_33(f, t0, y0, a, b, h)
        t_values, y_rk4 = runge_kutta_4(f, t0, y0, a, b, h)

        plt.plot(t_values, y_rk1, label=f'h={h:.2g}')
       # plt.plot(t_values, y_rk2, label=f'RK2 h={h:.2e}', alpha=0.5)
       # plt.plot(t_values, y_rk3, label=f'RK3 h={h:.2e}', alpha=0.5)
       # plt.plot(t_values, y_rk4, label=f'RK4 h={h:.2e}', alpha=0.5)

        y_exact = exact_solution(t_values)
        errors_rk1.append(np.max(np.abs(y_rk1 - y_exact)))
        errors_rk2.append(np.max(np.abs(y_rk2 - y_exact)))
        errors_rk3.append(np.max(np.abs(y_rk3 - y_exact)))
        errors_rk4.append(np.max(np.abs(y_rk4 - y_exact)))
        print(f"h={h:.2e}: RK1={errors_rk1[-1]:.2e}, RK2={errors_rk2[-1]:.2e}, \
              RK3={errors_rk3[-1]:.2e}, RK4={errors_rk4[-1]:.2e}")

    plt.plot(t_values, y_exact, label='Exact Solution', color='black', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('Runge-Kutta 1')
    #plt.legend()
    plt.grid()
    plt.show()

                    

    # Plotting the convergence
    plt.loglog(h_values, errors_rk1, marker='o', label='RK1 Error')
    plt.loglog(h_values, errors_rk2, marker='o', label='RK2 Error')
    plt.loglog(h_values, errors_rk3, marker='o', label='RK3 Error')
    plt.loglog(h_values, errors_rk4, marker='o', label='RK4 Error')

    plt.loglog(h_values, [h**1 for h in h_values], linestyle='--', label='O(h)')
    plt.loglog(h_values, [h**2 for h in h_values], linestyle='--', label='O(h^2)')
    plt.loglog(h_values, [h**3 for h in h_values], linestyle='--', label='O(h^3)')
    plt.loglog(h_values, [h**4 for h in h_values], linestyle='--', label='O(h^4)')
    
    plt.xlabel('Step size h')
    plt.ylabel('Max Error')
    plt.title('Convergence Study of RK Methods')
    plt.grid()
    plt.legend()
    plt.show()



def plot_solution(f, a, b, h, exact_solution):
    t0 = a
    y0 = 0
   
    t_values, y_rk1 = runge_kutta_1(f, t0, y0, a, b, h)
    t_values, y_rk2 = runge_kutta_2(f, t0, y0, a, b, h)
    t_values, y_rk3 = runge_kutta_3(f, t0, y0, a, b, h)
    t_values, y_rk4 = runge_kutta_4(f, t0, y0, a, b, h)


    # Exact solution for fine plotting
    t_fine = np.linspace(a, b, 100)
    y_ex_fine = exact_solution(t_fine)

    plt.plot(t_values, y_rk1, label='RK1 Method', 
             marker='o', markersize=3)
    plt.plot(t_values, y_rk2, label='RK2 Method', 
             marker='x', markersize=3)
    plt.plot(t_values, y_rk3, label='RK3 Method', 
             marker='s', markersize=3)
    plt.plot(t_values, y_rk4, label='RK4 Method', 
             marker='^', markersize=3)

    plt.plot(t_fine, y_ex_fine, label='Exact Solution', 
             linestyle='--')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title(f'h={h}')
    plt.legend()
    plt.grid()
    plt.show()
