from math import pi
import numpy as np


def kmatrices(k, P, Q, R, size, x=True, y=True, z=True):
    kv = []
    if x:
        p = np.arange(-P//2+1, P//2+1)
        kv.append(k[0] - 2 * pi * p / size[0])
    if y:
        q = np.arange(-Q//2+1, Q//2+1)
        kv.append(k[1] - 2 * pi * q / size[1])
    if z:
        r = np.arange(-R//2+1, R//2+1)
        kv.append(k[2] - 2 * pi * r / size[2])
    return (np.diag(e.flatten()) for e in np.meshgrid(*kv))

def nanometers(x):
    return x * 1e-9

def reciproc(a1, a2):
    coef = 2 * pi / (a1[0] * a2[1] - a1[1] * a2[0])
    b1 = (  a2[1] * coef, -a2[0] * coef)
    b2 = ( -a1[1] * coef,  a1[0] * coef)
    return b1, b2


