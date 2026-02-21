import numpy as np

def solve(p_f, samples):
    """
    Computes and solver the vandermonde matrix, and samples the resulting polynomial
    
    :param pf:          (x,y)-coordinates to construct vandermonde matrix from
    :param samples:     nr of sample points
    """
    v_matrix    = np.zeros((len(p_f), len(p_f)))
    y_vec       = p_f[:,1] 

    for i in range(0, len(p_f)):
        for j in range(0,len(p_f)):
            v_matrix[i][j] = p_f[i][0] ** j

    a_vec = np.linalg.solve(v_matrix, y_vec)

    step_size   = (p_f[-1][0] - p_f[0][0]) / samples
    points      = int((p_f[-1][0] - p_f[0][0]) / step_size)

    f = np.zeros((points + 1, 2))
    x = p_f[0][0]

    for i in range(0, len(f)):
        y   = 0
        pot = 0
        for a in a_vec:
            y   += a * (x ** pot)
            pot += 1
        f[i] = (x, y)
        x += step_size

    return f
