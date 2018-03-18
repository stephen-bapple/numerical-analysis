import numpy as np
from matplotlib import pyplot as plt

################################################################################
# Minimizers                                                                   #
################################################################################


def golden_section_search(f, a, b, tolerance=0.5e-08):
    g = (np.sqrt(5.0) - 1.0) / 2.0
    k = int(np.ceil(np.log(1.0 * tolerance / (b - a)) / np.log(g)))

    x1 = a + (1.0 - g) * (b - a)
    x2 = a + g * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    
    for i in range(k):
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + (1.0 - g) * (b - a)
            f2 = f1
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + g * (b - a)
            f1 = f2
            f2 = f(x2)
    return (a + b) / 2.0


def multivariate_newtons(J, H, W):
    s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))
    while np.linalg.norm(s) > tolerance:
        W += s
        s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))

    return W


def multivariate_newtons_multi_guess(J, H, starting_guesses):
    mins = []
    for W in starting_guesses:
        s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))
        while np.linalg.norm(s) > tolerance:
            W += s
            s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))

        mins.append(W)

    return mins


################################################################################
# Visualizers                                                                  #
################################################################################

def plot2d_with_mins(f, a, b, mins=[]):
    '''
    Plot a 2D function.
    '''
    
    x_pts = np.linspace(a, b, 1000)
    fig, ax = plt.subplots()
    ax.plot(x_pts, f(x_pts), color='blue')
    for min in mins:
        ax.scatter(min, f(min), color='red')
    plt.show()


def plot3d_with_mins(F, x_range=[-2, 2], y_range=[-2, 2], mins=[]):
    '''
    Plot a 2D function.
    '''
    
    n = 1000
    xmin = x_range[0]
    xmax = x_range[1]
    ymin = y_range[0]
    ymax = y_range[1]

    x = np.linspace(xmin, xmax, n + 1)
    y = np.linspace(ymin, ymax, n + 1)

    Z = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            Z[i, j] = F(x[i], y[j])
            
    X, Y = np.meshgrid(x, y)

    fig = pyplot.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    p1 = ax1.plot_surface(X, Y, Z, cmap=cm.jet, alpha=0.9)

    for min in mins:
        ax1.scatter3D(min[0], min[1], F(min[0], min[1]))

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    pyplot.show()


################################################################################
# Sentinel, for demoing.                                                       #
################################################################################

if __name__ == '__main__':
    # Demo golden section search:
    def f(x):
        return -1 * np.sin(np.pi * x)
    def g(x):
        return x**4 + 3 * x**3 + 9 * x
    a, b = -0.35, 1.3
    min = golden_section_search(f, a, b, 0.5e-10)
    
    print("The minimum is: %.10f" % min)
    
    plot2d_min(f, a, b, min)
