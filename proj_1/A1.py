import matplotlib.pyplot as plt
import numpy as np
import vandermonde_solver as vand_solver

p_f = np.array([[-3, 2], [-2.5, 1], [-2, 0.65], [0, 0.5], [2, 0.65], [2.5, 1], [3, 2]])
p_g = np.array([[-3, 2], [-2.5, 1.5], [-2, 1.2], [0, 1], [2, 1.2], [2.5, 1.5], [3, 2]])

f = vand_solver.solve(p_f, 100)
g = vand_solver.solve(p_g, 100)

plt.plot(f[:,0], f[:,1], color='red', label='f')
plt.plot(g[:,0], g[:,1], color='blue', label='g')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
