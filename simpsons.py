def integrate(f, samples, b, a):
    """
    Definite integral of f over a to b, using composite simpsons
    
    :param f:       the function to integrate 
    :param samples: nr of samples used in interpolation
    :param b:       upper bound
    :param a:       lower bound
    """
    if samples % 2:
        print("need even number of samples")
        return
    
    h   = (b - a) / (samples)
    n   = samples + 1
    sum = 0

    for i in range(1, int(n) - 1):
        if i % 2:   sum += 4 * f(i * h + a)
        else:       sum += 2 * f(i * h + a)

    return (f(a) + sum + f(b)) * (h / 3)
