import numpy as np
from matplotlib import pyplot

def F(w, h):
    return [ (w[i-1] - 2 * w[i] + w[i + 1]) / h**2 + np.exp(w[i]) for i in range(1, n)]

def J(w, h):
    od = ( 1 / h**2) * np.ones((len(w) - 1, 1))
    return np.diagflat(-2 / h**2 + np.exp(w))\
         + np.diagflat(od, 1)\
         + np.diagflat(od, -1) 


n = 50
alpha = 50
tol = 0.5e-08
I = [0, 1]
h = (I[-1] -  I[0]) / n
x = np.linspace(I[0], I[-1], n + 1)

for alpha in [0, 50]:
    w = [alpha * x[i] * (1 - x[i]) for i in range(n + 1)]
    
    w[0] = 0; w[-1] = 0 # Set boundary conditions

    print(J(w, h))

    #--------------------------------------------------------------------------#
    # Want to stop at Forward Error
    #--------------------------------------------------------------------------#

    s = np.linalg.solve(-1 * J(w[1:-1], h), F(w, h))
    counter = 0
    while np.linalg.norm(s) > tol:
        counter += 1
        w[1:-1] += s
        s = np.linalg.solve(-1 * J(w[1:-1], h), F(w, h))

    print(w)
    print(counter)

    pyplot.plot(x, w)
pyplot.show()
