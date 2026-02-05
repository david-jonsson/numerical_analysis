import matplotlib.pyplot as plt
import numpy as np
import lagrange_p as lag   
import simpsons as sim
import math
import boat

def f(x):
    return math.sin(21 * x) * math.e ** ((-0.3) * x) + x

samples = 1001
p_i = np.zeros((samples, 2))
for i in range(0, len(p_i)):
    x      = i / (samples - 1) * math.pi
    p_i[i] = (x, f(x))
    
print(sim.integrate(p_i))

# p_f = np.array([[-3, 2], [-2.5, 1], [-2, 0.65], [0, 0.5], [2, 0.65], [2.5, 1], [3, 2]])
# p_g = np.array([[-3, 2], [-2.5, 1.5], [-2, 1.2], [0, 1], [2, 1.2], [2.5, 1.5], [3, 2]])

p_f, p_g = boat.boat()

step_size = (p_f[-1][0] - p_f[0][0]) / 100000

f = lag.get_lagrange_p(step_size, p_f[:,0], p_f[:,1])
g = lag.get_lagrange_p(step_size, p_g[:,0], p_g[:,1])

h_numerator = np.zeros((len(f), 2))
h_denominator = np.zeros((len(g), 2))

for i in range(0, len(h_numerator)):
    h_numerator[i]   = (f[i][0], (f[i][1] - g[i][1]) * f[i][0])
    h_denominator[i] = (f[i][0], f[i][1] - g[i][1])

xc_num  = sim.integrate(h_numerator)
xc_den  = sim.integrate(h_denominator)
xc      = xc_num / xc_den

print(xc_num / xc_den)



i_num = np.zeros((len(f), 2))
i_den = np.zeros((len(g), 2))

for i in range(0, len(i_num)):
    i_num[i] = (g[i][0], ((f[i][1] ** 2) - (g[i][1] ** 2)) * (1 / 2))
    i_den[i] = (g[i][0], f[i][1] - g[i][1])

yc_num  = sim.integrate(i_num)
yc_den  = sim.integrate(i_den)
yc      = yc_num / yc_den

print(yc_num / yc_den)


plt.plot(f[:,0], f[:,1], color='red', label='f')
plt.plot(g[:,0], g[:,1], color='blue', label='g')
plt.plot(xc, yc, 'o')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
