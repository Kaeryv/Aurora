from math import pi
import numpy as np


def kmatrices(k, P, Q, R, size):
    p = np.arange(-P//2+1, P//2+1)
    q = np.arange(-Q//2+1, Q//2+1)
    r = np.arange(-R//2+1, R//2+1)

    bx = k[0] - 2 * pi * p / size[0]
    by = k[1] - 2 * pi * q / size[1]
    bz = k[2] - 2 * pi * r / size[2]
    return (np.diag(e.flatten()) for e in np.meshgrid(bx, by, bz))


