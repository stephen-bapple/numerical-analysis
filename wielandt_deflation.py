import numpy as np

def power_iteration(A, v0, tol=0.5e-8):
    print(A)
    print(v0)
    v = v0
    while np.linalg.norm(A*v - float(v.T * A * v) * v) > tol:
        v = A * v
        v = v / np.linalg.norm(v)

    return v

A = np.matrix([[1, 1, 0, 0],
               [1, 2, 0, 1],
               [0, 0, 3, 3],
               [0, 1, 3, 2]])
v0 = np.matrix([[1], [1], [0], [1]])
lam = []

v = power_iteration(A, v0)
i = np.argmax(v)

print("Eigvect: ", v)
lam.append(float(v.T * A * v))
print("Eigval: ", lam[0])

print(i)
print(v[i])


x = (1 / (lam[0] * float(v[i]))) * A[i, :].T
B = A - lam[0] * v * x.T
print(B)

B = np.delete(np.delete(B, i, 0), i, 1)
print(B)
