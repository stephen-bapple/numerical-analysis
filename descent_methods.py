import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 
def F(x):
    return x[0]**4 + x[1]**4 + 2*x[0]**2 * x[1]**2 + 6*x[0]*x[1] - 4 * x[0] - 4 * x[1] + 1
    
def J(x):
    return [4*x[0]**3 + 4 * x[0] * x[1]**2 + 6*x[1] - 4, 4*x[1]**3 + 4*x[0]**2*x[1] + 6*x[0] - 4]

def H(x, y):
    return [[12*x**2 + 4 * y**2, 8*x*y + 6], [8 * x * y + 6, 12*y**2 + 4*x**2]]

def plot():    
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
            Z[i, j] = F([x[i], y[j]])
            
    X, Y = np.meshgrid(x, y)

    fig = pyplot.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    p1 = ax1.plot_surface(X,Y,Z, cmap=cm.jet)
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.show()

def find_min():
    # I have to find the minimums

    #starting values
    W = [-1, 1]
    y = 0

    for x in [[-1, 1],[1, -1]]:
        
        s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))
        W += s
    print(W)

if __name__ == '__main__':
    find_min()