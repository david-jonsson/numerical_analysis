import math

def integrate(f):
    """
    Definite integral of f over a to b, using composite simpsons
    
    :param f:       the function to integrate
    """

    nr_samples  = len(f)
    nr_segments = len(f) - 1
    h           = (f[-1][0] - f[0][0]) / nr_segments
    sum         = 0

    if nr_samples % 2 == 0:
        print("need odd number of samples")
        return

    for i in range(1, nr_segments):
        if i % 2:   sum += 4 * f[i][1]
        else:       sum += 2 * f[i][1]

    return (f[0][1] + sum + f[-1][1]) * (h / 3)
