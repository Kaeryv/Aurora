from aurora.convolution import convmat
from aurora.misc import kmatrices, kmatrices
import numpy as np
from scipy.linalg import eig
from math import pi
import matplotlib.pyplot as plt
N = 301
x = np.arange(-N//2+1, N//2+1)
X, Y = np.meshgrid(x, x)
struct = np.ones((N, N, 1))
struct[X**2+Y**2<(0.2*N)**2] = 8.9
P, Q, R = 7, 7, 1
RC = convmat(struct, P, Q)
UC = convmat(np.ones((N, N, 1)), P, Q)
omegasE= []
omegasH= []
for i, beta in enumerate(np.linspace(0, 0.5, 40)):
    Kx, Ky, Kz = kmatrices([0, beta * 2 * pi / 1.0, 0], P, Q, R, (1.0,1.0,0.0))
    # E mode
    A = Kx @ np.linalg.inv(UC) @ Kx  + Ky @ np.linalg.inv(UC) @ Ky
    w, v = eig(A, RC)
    k0 = w
    omega = np.real(np.sqrt(k0) / 2 / pi)
    omegasE.append(omega)
    # H mode
    A = Kx @ np.linalg.inv(RC) @ Kx  + Ky @ np.linalg.inv(RC) @ Ky
    w, v = eig(A, UC)
    k0 = w
    omega = np.real(np.sqrt(k0) / 2 / pi)
    omegasH.append(omega)

SZ = v[:, -1].reshape(7,7)
EZ =  np.fft.ifft2(np.fft.ifftshift(np.pad(SZ,63)))
extent = [0,2,0,2]
plt.matshow(np.tile(np.real(EZ), (2,2)), cmap="bwr", extent=extent)
plt.contour(np.tile(np.real(struct[...,0]), (2,2)), color="k",extent=extent)
plt.axis("equal")
plt.show()

omegasE = np.asarray(omegasE)
omegasH = np.asarray(omegasH)


# Plotting
import matplotlib.pyplot as plt
for o in omegasE.T:
    plt.plot(o, 'b.')
for o in omegasH.T:
    plt.plot(o, 'r.')
plt.axis([0, 40, 0, 1])
plt.show()
# plt.matshow(np.abs(UC))
# plt.show()
# plt.matshow(np.abs(Kx))
# plt.show()
