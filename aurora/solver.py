from tqdm import tqdm
import numpy as np
from aurora.misc import kmatrices, nanometers, reciproc
from aurora.convolution import convmat

from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from math import pi

class PWE3D:
    def __init__(self):
        pass
    
    def solve_kpoints(self, path, struct, P, Q, R, size=(1.0, 1.0, 1.0), num_bands=10):
        RC = convmat(struct, P, Q, R)
        UC = convmat(np.ones_like(struct), P, Q, R)
        omegas = []
        for i, (bx, by) in enumerate(tqdm(path)):
            Kx, Ky, Kz = kmatrices([bx* 2 * pi, by * 2 * pi / 1.0, 0], P, Q, R, size)
            P12 = np.zeros((len(np.diag(Kx)), 2, 3))
            for i, (kx, ky, kz) in enumerate(zip(np.diag(Kx), np.diag(Ky), np.diag(Kz))):
                kv = np.array([kx, ky, kz])
                if np.linalg.norm(kv) < 1e-14:
                    p1 = np.array([0, 0, 1])
                    p2 = np.array([0, 1, 0])
                else:
                    iv = np.array([4*ky, 2*kz, 3*kx])
                    p1 = np.cross(kv, iv)
                    p1 /= np.linalg.norm(p1)
                    p2 = np.cross(kv, p1)
                    p2 /= np.linalg.norm(p2)
                P12[i, 0, :] = p1
                P12[i, 1, :] = p2

            P1 = np.hstack((np.diag(P12[:, 0, 0]), np.diag(P12[:, 0, 1]), np.diag(P12[:, 0, 2])))
            P2 = np.hstack((np.diag(P12[:, 1, 0]), np.diag(P12[:, 1, 1]), np.diag(P12[:, 1, 2])))

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
            KIMRCK = K @ IMRC  @ K
            Aul = P1 @ KIMRCK @ P1.T
            Aur = P1 @ KIMRCK @ P2.T
            All = P2 @ KIMRCK @ P1.T
            Alr = P2 @ KIMRCK @ P2.T
            A = np.vstack((
                np.hstack((Aul, Aur)),
                np.hstack((All, Alr)),
            ))
            # # E mode
            w, v = eigs(A, k=num_bands, which="SM")
            #w, v = np.linalg.eig(A)
            k0 = w
            omega = np.real(np.sqrt(-k0) / 2 / pi)
            omegas.append(omega)
        omegas = np.asarray(omegas)

        return omegas
    
class PWE2D:
    def __init__(self):
        pass
    
    def solve_kpoints(self, struct, path, pw, size):
        RC = convmat(struct, *pw)
        UC = convmat(np.ones_like(struct), *pw)
        omegasE= []
        omegasH= []
        for i, (b1, b2) in enumerate(path):
            Kx, Ky = kmatrices([b1 * 2 * pi, b2 * 2 * pi, 0], *pw, 1, size, z=False)
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

        omegasE = np.asarray(omegasE)
        omegasH = np.asarray(omegasH)
        
        return omegasE, omegasH
