import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Function
def F(x, y):
    return x**4 + y**4 + 2 * x**2 * y**2 + 6 * x * y - 4 * x - 4 * y + 1

# Jacobian
def J(x, y):
    return [4 * x**3 + 4 * x * y**2 + 6 * y - 4, 
            4 * y**3 + 4 * x**2 * y + 6 * x - 4]

# Hessian
def H(x, y):
    return [[12*x**2 + 4 * y**2, 8*x*y + 6],
            [8 * x * y + 6, 12*y**2 + 4*x**2]]

# 3D Plot
def plot(F, minimums=[]):    
    n = 1000
    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    x = np.linspace(xmin, xmax, n + 1)
    y = np.linspace(ymin, ymax, n + 1)

    Z = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            Z[i, j] = F(x[i], y[j])
            
    X, Y = np.meshgrid(x, y)

    fig = pyplot.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    p1 = ax1.plot_surface(X, Y, Z, cmap=cm.jet, alpha=0.8)
    
    for min in minimums:
        ax1.scatter3D(min[0], min[1], F(min[0], min[1]))

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    pyplot.show()

    
        
def find_min(J, H, tolerance=0.5e-8):
    mins = []
    for W in [[-1, 1], [1, -1]]:
        s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))
        while np.linalg.norm(s) > tolerance:
            W += s
            s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))

        print(W)
        mins.append(W)
    return mins
    
if __name__ == '__main__':
    mins = find_min(J, H)
    plot(F)#, mins)