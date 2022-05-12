import numpy as np

def convmat(A, P, Q=1, R=1):
    nx, ny, nz = A.shape
    nh = P * Q * R
    p = np.arange(-P//2+1, P//2+1)
    q = np.arange(-Q//2+1, Q//2+1)
    r = np.arange(-R//2+1, R//2+1)

    A = np.fft.fftshift(np.fft.fftn(A)) / nx / ny / nz
    p0 = int(np.floor(nx/2))
    q0 = int(np.floor(ny/2))
    r0 = int(np.floor(nz/2))

    C = np.zeros((nh, nh), dtype=np.complex128)
    for rrow in range(R):
        for qrow in range(Q):
            for prow in range(P):
                row = rrow * Q * P + qrow * P + prow
                for rcol in range(R):
                    for qcol in range(Q):
                        for pcol in range(P):
                            col = rcol * P * Q + qcol * P + pcol
                            pfft = p[prow] - p[pcol]
                            qfft = q[qrow] - q[qcol]
                            rfft = r[rrow] - r[rcol]
                            C[row, col] = A[p0+pfft, q0+qfft, r0+rfft]
    
    return C
