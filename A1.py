import matplotlib.pyplot as plt
import numpy as np
import lagrange_p as lag   
import simpsons as sim

p_f = np.array([[-3, 2], [-2.5, 1], [-2, 0.65], [0, 0.5], [2, 0.65], [2.5, 1], [3, 2]])
p_g = np.array([[-3, 2], [-2.5, 1.5], [-2, 1.2], [0, 1], [2, 1.2], [2.5, 1.5], [3, 2]])

step_size = (p_f[-1][0] - p_f[0][0]) / 100

f = lag.get_lagrange_p(step_size, p_f[:,0], p_f[:,1])
g = lag.get_lagrange_p(step_size, p_g[:,0], p_g[:,1])

h_num = np.zeros((len(f), 2))
h_den = np.zeros((len(g), 2))

plt.plot(f[:,0], f[:,1], color='red', label='f')
plt.plot(g[:,0], g[:,1], color='blue', label='g')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
