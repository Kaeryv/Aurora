from argparse import ArgumentParser
parser = ArgumentParser()
# parser.add_argument("-pw", nargs='+', required=True, type=int)
parser.add_argument("-conf", type=int, required=True)
args = parser.parse_args()
assert len(args.pw) == 3
from cmath import nan
from pkgutil import extend_path
from aurora.convolution import convmat
from aurora.misc import kmatrices, kmatrices
import numpy as np
from scipy.linalg import eig, eigvals
from math import pi
#import matplotlib.pyplot as plt
from aurora.epsilon import Layer
from aurora.math import rotation_matrix
from numpy.matlib import repmat
from math import sin, cos
N = 301
def nanometers(x):
    return x * 1e-9
a0 = nanometers(478)
e = nanometers(131) / a0
X = np.arange(1, 15, 2)
from itertools import product
confs = list(product(X, X, X))
layer = Layer(a1=[1, 0], a2=[0, 1], resolution=(N, N), supersample=1, size=(1, 2*sin(pi/3)))
layer.draw_block((0.5, 0), (2.0, e), theta=-pi/3)
layer.draw_block((0.5, 0), (2.0, e), theta=pi/3)
layer.draw_block((0.5, 2*sin(pi/3)), (2.0, e), theta=pi/3)
layer.draw_block((0.5, 2*sin(pi/3)), (2.0, e), theta=-pi/3)
layer.draw_block((0.5, 0.866), (2*0.866, e),e1=[1., 0.], e2=[0., 1.], theta=0)
layer.draw_block((0.5, 0), (2*0.866, e),e1=[1., 0.], e2=[0., 1.], theta=0)
layer.draw_block((0.5, 2*0.866), (2*0.866, e),e1=[1., 0.], e2=[0., 1.], theta=0)
#plt.matshow(repmat(layer.epsilon, 3, 3), extent=[0, 1, 0, 2*0.86])
#plt.show()
x = np.arange(-N//2+1, N//2+1)
eps_si = 3.48**2
struct  = np.ones((N, N, N))
center = N//2
h = int(nanometers(220) / a0 * N)
for i in range(center, center+h):
    struct[:, :, i] = 1.0 + layer.epsilon * (eps_si-1)
P, Q, R = confs[args.conf]
RC = convmat(struct, P, Q, R)
UC = convmat(np.ones((N, N, N)), P, Q, R)
omegas = []
from tqdm import tqdm
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix
def freqs_at_kpoint(beta):
    Kx, Ky, Kz = kmatrices(beta, P, Q, R, [1.0, 1.0, 1.0])
    P1x = np.zeros_like(Kx)
    P1y = np.zeros_like(Kx)
    P1z = np.zeros_like(Kx)
    P2x = np.zeros_like(Kx)
    P2y = np.zeros_like(Kx)
    P2z = np.zeros_like(Kx)
    for i, (kx, ky, kz) in enumerate(zip(np.diag(Kx), np.diag(Ky), np.diag(Kz))):
        kv = np.array([kx, ky, kz])
        if np.linalg.norm(kv) < 1e-8:
            p1 = np.array([0, 0, 1])
            p2 = np.array([0, 1, 0])
        else:
            iv = np.array([4*ky, 2*kz, 3*kx])
            p1 = np.cross(kv, iv)
            p1 /= np.linalg.norm(p1)
            p2 = np.cross(kv, p1)
            p2 /= np.linalg.norm(p2)
        P1x[i, i] = p1[0]
        P1y[i, i] = p1[1]
        P1z[i, i] = p1[2]
        P2x[i, i] = p2[0]
        P2y[i, i] = p2[1]
        P2z[i, i] = p2[2]
    P1 = np.hstack((P1x, P1y, P1z))
    P2 = np.hstack((P2x, P2y, P2z))


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
    IMRC = np.linalg.inv(MRC)
    Aul = P1 @ K @ IMRC  @ K @ P1.T
    Aur = P1 @ K @ IMRC  @ K @ P2.T
    All = P2 @ K @ IMRC  @ K @ P1.T
    Alr = P2 @ K @ IMRC  @ K @ P2.T
    A = np.vstack((
        np.hstack((Aul, Aur)),
        np.hstack((All, Alr)),
    ))
    A = csc_matrix(A)
    # # E mode
    #k0 = eigvals(A)
    k0 = eigs(A, 5, which="SM", tol=1e-4, return_eigenvectors=False)
    omega = np.real(np.sqrt(-k0) / 2 / pi)
    return omega
# Gamma - K
for i, beta in enumerate(tqdm(np.linspace(0, 1/3, 20))):
    omegas.append(freqs_at_kpoint([ -beta * 2 * pi / 1.0, beta * 2 * pi / 1.0, 0]))
for i, (beta1, beta2) in enumerate(tqdm(list(zip(np.linspace(-1/3,  0.0, 20), np.linspace(1/3, 0.5, 20))))):
    omegas.append(freqs_at_kpoint([ beta1 * 2 * pi, beta2 * 2 * pi , 0]))
omegas = np.asarray(omegas)
np.savez_compressed(f"freqs_{args.conf}.npy", freqs=omegas, P=P, Q=Q, R=R)
# from scipy.constants import c
# # Plotting
# for o in omegas.T:
#     plt.plot(o / a0 * c / 1e12, 'b.')
# #plt.axis([0, 20, 160, 220])
# plt.show()
# plt.matshow(np.abs(C))
# plt.show()
# plt.matshow(np.abs(Kx))
# plt.show()