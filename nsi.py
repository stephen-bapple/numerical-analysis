# Normalized simultaneous iteration NSI
# LIE: actually QR iteration
N = 160
import numpy as np

A = np.matrix([[3., 1., 2.],
                  [1., 3., -2.],
                  [2., -2., 6.]])

                  
def qr_iteration(A):
    Q, R = np.linalg.qr(A)
    for k in range(N):
        Q, R = np.linalg.qr(R * Q)

    return R*Q


def nsi(A):
    Q = np.identity(A.shape[0])
    
    for k in range(N):
        Q, R = np.linalg.qr(A*Q)
    
    lambd = Q.T * A * Q
    
    return lambd

    
print(qr_iteration(A))
print('----------------------')
print(nsi(A))
print('----------------------')
print(np.linalg.eig(A))
