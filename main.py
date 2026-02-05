import matplotlib.pyplot as plt
import numpy as np
import lagrange_p as lag   
import simpsons as sim
import math

def f(x):
    return math.sin(21 * x) * math.e ** ((-0.3) * x) + x

samples = 1001
p_i = np.zeros((samples, 2))
for i in range(0, len(p_i)):
    x      = i / (samples - 1) * math.pi
    p_i[i] = (x, f(x))
    
print(sim.integrate(p_i))

p_f = np.array([[-3, 2], [-2.5, 1], [-2, 0.65], [0, 0.5], [2, 0.65], [2.5, 1], [3, 2]])
p_g = np.array([[-3, 2], [-2.5, 1.5], [-2, 1.2], [0, 1], [2, 1.2], [2.5, 1.5], [3, 2]])
step_size = 0.1

f = lag.get_lagrange_p(step_size, p_f[:,0], p_f[:,1])
g = lag.get_lagrange_p(step_size, p_g[:,0], p_g[:,1])

xc = np.zeros((len(f), 2))
yc = np.zeros((len(g), 2))

for i in range(0, len(xc)):
    numerator   = (f[i][1] - g[i][1]) * f[i][0]
    denominator = f[i][1] - g[i][1]
    if denominator == 0: 
        xc[i]       = (f[i][0], 0)
        continue
    xc[i]       = (f[i][0], numerator / denominator)

for i in range(0, len(yc)):
    numerator   = ((f[i][1] ** 2) - (g[i][1] ** 2)) * (1 / 2)
    denominator = f[i][1] - g[i][1]
    if denominator == 0: 
        yc[i]       = (g[i][0], 0)
        continue
    yc[i]       = (g[i][0], numerator / denominator)
    print(yc[i])

print(yc)

print(sim.integrate(xc))
print(sim.integrate(yc))



# plt.plot(f[:,0], f[:,1], color='red', label='f')
# plt.plot(g[:,0], g[:,1], color='blue', label='g')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
