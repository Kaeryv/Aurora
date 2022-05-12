from aurora.convolution import convmat
from aurora.misc import kmatrices, kmatrices
import numpy as np
from scipy.linalg import eig
from math import pi
import matplotlib.pyplot as plt

N = 201
x = np.arange(-N//2+1, N//2+1)
X, Y, Z = np.meshgrid(x, x, x)
struct = np.ones((N, N, N))
struct[X**2+Y**2+Z**2<(0.2*N)**2] = 8.9
plt.matshow(struct[100])
plt.show()
P, Q, R = 13, 13, 13
RC = convmat(struct, P, Q, R)
UC = convmat(np.ones((N, N, N)), P, Q, R)
omegas = []
from tqdm import tqdm
for i, beta in enumerate(tqdm(np.linspace(0, 0.5, 20))):
    Kx, Ky, Kz = kmatrices([0, beta * 2 * pi / 1.0, 0], P, Q, R, 1.0)
    ZERO = np.zeros_like(Kz)
    K = np.vstack([
        np.hstack(( ZERO, -Kz,    Ky)),
        np.hstack(( Kz,    ZERO, -Kx)),
        np.hstack((-Ky,    Kx,    ZERO)),
    ])
    MRC = np.vstack([
        np.hstack(( RC,    ZERO,    ZERO)),
        np.hstack(( ZERO,    RC,    ZERO)),
        np.hstack(( ZERO,    ZERO,    RC)),
    ])
    # E mode
    w, v = eig(K @ K, MRC)
    k0 = w
    omega = np.real(np.sqrt(k0) / 2 / pi)
    omegas.append(omega)

omegas = np.asarray(omegas)


# Plotting
for o in omegas.T:
    plt.plot(o, 'b.')
#plt.axis([0, 40, 0, 1])
plt.show()
# plt.matshow(np.abs(C))
# plt.show()
# plt.matshow(np.abs(Kx))
# plt.show()