def integrate(f, segments, b, a):
    if segments % 2:
        print("need ever number of segments")
        return
    
    h =  (b - a) / (segments)
    n = segments + 1
    sum = 0

    for i in range(1, int(n) - 1):
        if i % 2:
            sum += 4 * f(i * h + a)
        else:
            sum += 2 * f(i * h + a)

    return (f(a) + sum + f(b)) * (h / 3)
