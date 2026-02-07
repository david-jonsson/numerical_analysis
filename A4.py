import matplotlib.pyplot as plt
import numpy as np
import lagrange_p as lag   
import simpsons as sim


def boat():
    p_f = np.array([
    [98.7,   52.9],
    [119.4,  49.4],
    [140.4,  46.4],
    [165.1,  44.1],
    [194.2,  42.5],
    [219.1,  42.1],
    [237.8,  60.1],
    [292.8,  60.5],
    [302.1,  60.4],
    [453.4,  61.2],
    [456.5,  48.7],
    [467.6,  40.3],
    [541.5,  40.1],
    [590.4,  42.7],
    [631.1,  49.4]
    ])

    p_g = np.array([
    [98.7,   52.9],
    [146.1,  -0.5],
    [158.8, -14.0],
    [290.4, -26.5],
    [337.6, -93.8],
    [385.8, -93.1],
    [386.1, -29.2],
    [560.2, -15.1],
    [565.2, -60.0],
    [575.2, -60.0],
    [580.4,  -6.5],
    [587.6,   0.0],
    [616.8,  23.3],
    [631.1,  49.4]
    ])
    
    return p_f, p_g

p_f, p_g = boat()

step_size = (p_f[-1][0] - p_f[0][0]) / 100


x = np.linspace(p_f[0][0], p_f[-1][0], 2001)

f = lag.func_pw_linear(x, p_f[:,0], p_f[:,1])
g = lag.func_pw_linear(x, p_g[:,0], p_g[:,1])

h_num = np.zeros((len(f), 2))
h_den = np.zeros((len(g), 2))

for i in range(0, len(h_num)):
    h_num[i] = (x[i], (f[i] - g[i]) * x[i])
    h_den[i] = (x[i], f[i] - g[i])

xc_num  = sim.integrate(h_num)
xc_den  = sim.integrate(h_den)
xc      = xc_num / xc_den

print(xc)

i_num = np.zeros((len(f), 2))
i_den = np.zeros((len(g), 2))

for i in range(0, len(i_num)):
    i_num[i] = (x[i], ((f[i] ** 2) - (g[i] ** 2)) * (1 / 2))
    i_den[i] = (x[i], f[i] - g[i])

yc_num  = sim.integrate(i_num)
yc_den  = sim.integrate(i_den)
yc      = yc_num / yc_den

print(yc)


plt.plot(x, f, color='red', label='f')
plt.plot(x, g, color='blue', label='g')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.plot(x, f, color='red', label='f')
plt.plot(x, g, color='blue', label='g')
plt.plot(xc, yc, 'o')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
