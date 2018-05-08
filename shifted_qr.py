# Shifted QR iteration. 
# 
# This process extracts eigenvalues from a non-symmetric matrix that may contain
# complex eigenvalues.

import numpy as np
                  
def shifted_qr(A, max_iter, tol):
    eigenvalues = []
    
    # Iterate until the matrix is a single value or empty.
    while A.shape[0] > 1:
        k = 0
        n = A.shape[0]
        I = np.eye(n)

        while np.linalg.norm(A[n - 1, :-1]) > tol and k < max_iter:
            s = A[n - 1, n - 1]
            Q, R = np.linalg.qr(A - s * I)
            A = R * Q + s * I
            k += 1

        # Check if we need to extract complex values.
        if k == max_iter and np.linalg.norm(A[n - 1, :-1]) > tol:
            # Need to find complex eigens
            eg = np.linalg.eig(A)
            eigenvalues.append(eg[0][0])
            eigenvalues.append(eg[0][1])

        else: # Have 1 real eigenvalue.
            eigenvalues.append(A[n-1,n-1])

        # Deflate the matrix.
        A = A[:n-1, :n-1]

    # Return all the eigenvalues.
    return eigenvalues
    
if __name__ == "__main__":
    A = np.matrix([[3., -1., -2.],
                   [3., 2., -3.],
                   [1., 2., 0]])

    eigenvalues = shifted_qr(A, 100, 0.5e-10)
    print(eigenvalues)
    print("vs")
    print(np.linalg.eig(A))
