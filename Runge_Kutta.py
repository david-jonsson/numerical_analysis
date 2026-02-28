import numpy as np
import matplotlib.pyplot as plt

h = 0.1

k = 0.001
a = 0.02
g = 9.82

angle = np.deg2rad(40)
v_0 = 40
v_x = v_0 * np.cos(angle)
v_y = v_0 * np.sin(angle)


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




#------------------------------------------------------------
# Define the rhs of the differential equation dy/dt = f(t)
#------------------------------------------------------------
def f(t, u):
    """Defines the Lorenz system of ODEs."""
    k = 0.001
    a = 0.02
    g = 9.82
    dudt = np.array([
        u[1],
        -k * u[1] + a * np.sin(t),
        u[3],
        -k * u[3] - g
    ])
    return dudt

#------------------------------------------------------------
# Exact solution for comparison
#------------------------------------------------------------
def exact_solution(t):

    y_values = []
    for i in t:
        y_values.append(np.array([x_solution(i), y_solution(i)]))
    return np.array(y_values)

#------------------------------------------------------------
# Runge-Kutta 1nd order method
#------------------------------------------------------------
def runge_kutta_1(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = [y0.copy()]
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        y0 += h * k1
        t0 += h
        print(t0, k1, y0)
        t_values.append(t0)
        y_values.append(y0.copy())
    print(y_values)
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
    y_values = [y0.copy()]
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + h / 3, y0 + (h / 3) * k1)
        k3 = f(t0 + 2/3*h, y0 + 2/3*h * k2)
        y0 += h *(1/4 * k1 + 3/4 * k3)
        t0 += h
        t_values.append(t0)
        y_values.append(y0.copy())

    return np.array(t_values), np.array(y_values)


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
    
    return np.array(t_values), np.array(y_values)

#------------------------------------------------------------
# convergence study
#------------------------------------------------------------
def convergence_study():
    a = 0
    b = 2 * np.pi

    # h_values = [1, 0.5, 0.25, 0.125]

    h_values = [1, 0.5, 0.25]
    # h_values = [b/n for n in [5, 10, 20, 40, 80, 160, 320, 640]]
    errors_rk4 = []

    plt.figure()
    for h in h_values:

        t0 = a
        y0 = np.array([0, v_x, 0, v_y])

        t_values, y_rk4 = runge_kutta_4(f, t0, y0, a, b, h)
        y_rk4 = np.array(y_rk4[:, [0, 2]])
                          
        y_exact = exact_solution(t_values)
        errors_rk4.append(np.average(np.abs(y_rk4 - y_exact)))
        print(f"h={h:.2e}: RK4={errors_rk4[-1]:.2e}")
        print("estimate:")
        print(y_rk4)
        print("exact:")
        print(y_exact)
        # print(y_rk4.shape)
        # print(y_exact.shape)
        # print(h, np.abs(y_rk4 - y_exact))


    plt.plot(t_values, y_exact, label='Exact Solution', color='black', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('Runge-Kutta 1')
    #plt.legend()
    plt.grid()
    plt.show()
                    
    plt.loglog(h_values, errors_rk4, marker='o', label='RK4 Error')
    plt.loglog(h_values, [h**4 for h in h_values], linestyle='--', label='O(h^4)')
    plt.xlabel('Step size h')
    plt.ylabel('Max Error')
    plt.title('Convergence Study of RK Methods')
    plt.grid()
    plt.legend()
    plt.show()



#------------------------------------------------------------
# Main execution
#------------------------------------------------------------
if __name__ == "__main__":
    
    a = 0
    b = 2 * np.pi
    t0 = a
    y0 = np.array([0, v_x, 0, v_y])
    h = 0.5
   
    t_values, y_rk4 = runge_kutta_4(f, t0, y0, a, b, h)
    y_rk4 = np.array(y_rk4[:, [0, 2]])


    # convergence study
    convergence_study()


    # Exact solution for fine plotting
    t_fine = np.linspace(a, b, 100)
    y_ex_fine = exact_solution(t_fine)

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
