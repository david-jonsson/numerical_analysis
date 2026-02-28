import numpy

def lorenz_system(t, u):
    """Defines the Lorenz system of ODEs."""
    k = 0.001
    a = 0.02
    g = 9.82
    dudt = np.array([
        u[1],
        -k * u[1] + a * numpy.sin(t),
        u[3],
        -k * u[1] - g
    ])
    return dudt