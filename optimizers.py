import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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


def multivariate_newtons(J, H, W, tolerance=0.5e-8):
    s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))
    while np.linalg.norm(s) > tolerance:
        W += s
        s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))

    return W


def multivariate_newtons_multi_guess(J, H, starting_guesses, tolerance=0.5e-8):
    mins = []
    for W in starting_guesses:
        s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))
        while np.linalg.norm(s) > tolerance:
            W += s
            s = np.linalg.solve(np.multiply(-1, H(W[0],W[1])), J(W[0], W[1]))

        mins.append(W)

    return mins

# Weakest line search with backtracking.
# Type of steepest_descent.
def weakest_line(F, J, x0, s_max=0.5, delta=1.0e-03, tolerance=0.5e-08):
    x = x0
    s = s_max
    v = np.multiply(-1, J(x[0], x[1]))
    
    while np.linalg.norm(np.multiply(s, v), 2) < tolerance:
        
        # No sense recomputing these every time...
        fx = F(x[0], x[1])
        jTv = np.multiply(np.transpose(J(x[0], x[1])), v)
        
        while F(x + np.multiply(s, v)) <= fx + delta * s * jTv:
            s /= 2.0

        x += np.multiply(s, v)
        v = np.multiply(-1, J(x[0], x[1]))
        
    return x

    
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

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    p1 = ax1.plot_surface(X, Y, Z, cmap=cm.jet, alpha=0.9)

    for min in mins:
        ax1.scatter3D(min[0], min[1], F(min[0], min[1]))

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.show()


################################################################################
# Sentinel, for demoing.                                                       #
################################################################################

if __name__ == '__main__':
    
    # Demo golden section search:
    ############################################################################
    def demo_gss():
        def f(x):
            return -1 * np.sin(np.pi * x)
        def g(x):
            return x**4 + 3 * x**3 + 9 * x
        a, b = -0.35, 1.3
        min = golden_section_search(f, a, b, 0.5e-10)
        
        print("The minimum is: %.10f" % min)
        
        plot2d_with_mins(f, a, b, [min])
    #demo_gss()
    
    # Demo multivariate Newtons
    ############################################################################
    def demo_mv_newton():
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
        print('Plotting both minimums...')
        starting_guesses = [[-1, 1], [1, -1]]
        mins = multivariate_newtons_multi_guess(J, H, starting_guesses)
        plot3d_with_mins(F, mins=mins)
        
        print('Plotting only one minimum...')
        starting_guess = [-1, 1]
        min = multivariate_newtons(J, H, starting_guess)
        plot3d_with_mins(F, mins=[min])
    #demo_mv_newton()
    
    def demo_weakest_line():
        def F(x, y):
            return x**4 + y**4 + 2 * x**2 * y**2 + 6 * x * y - 4 * x - 4 * y + 1

        # Jacobian
        def J(x, y):
            return [4 * x**3 + 4 * x * y**2 + 6 * y - 4, 
                    4 * y**3 + 4 * x**2 * y + 6 * x - 4]

        x0 = [-1, 1]
        min = weakest_line(F, J, x0)
        plot3d_with_mins(F, mins=[min])
    demo_weakest_linet()
    
    def demo_weakest_line_gss():
        pass
    demo_weakest_line_gss()

