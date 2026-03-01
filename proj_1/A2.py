import matplotlib.pyplot as plt
import numpy as np  
import simpsons as sim
import math

def f(x):
    return math.sin(21 * x) * math.e ** ((-0.3) * x) + x

def check_error(samples):
    p_i = np.zeros((samples, 2))
    for i in range(0, len(p_i)):
        x      = i / (samples - 1) * math.pi
        p_i[i] = (x, f(x))
    integral = sim.integrate(p_i)
    return abs(integral - float(5.000963038207926))
    
nr_samples = []

for i in range (4, 15):
    nr_samples.append(2 ** i + 1) 

errors = []
stepsizes = []
h4 = []

for i in nr_samples:
    stepsize = math.pi / i
    h4.append(stepsize ** 4)
    stepsizes.append(stepsize)
    errors.append(check_error(i))

plt.loglog(stepsizes, errors, color='red', label='error')
plt.loglog(stepsizes, h4, color='blue', label='h4')
plt.grid()
plt.xlabel('ln(h)')
plt.ylabel('ln(error)')
plt.legend()
plt.show()
