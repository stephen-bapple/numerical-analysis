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
    s = np.linalg.solve(np.multiply(-1, H(W)), J(W))
    while np.linalg.norm(s) > tolerance:
        W += s
        s = np.linalg.solve(np.multiply(-1, H(W)), J(W))

    return W


def multivariate_newtons_multi_guess(J, H, starting_guesses, tolerance=0.5e-8):
    mins = []
    for W in starting_guesses:
        s = np.linalg.solve(np.multiply(-1, H(W)), J(W))
        while np.linalg.norm(s) > tolerance:
            W += s
            s = np.linalg.solve(np.multiply(-1, H(W)), J(W))

        mins.append(W)

    return mins


# def weakest_line(F, J, x0, s_max=0.5, delta=1.0e-03, tolerance=0.5e-08):
#     """
#     Steepest descent method.
#     Weakest line search with backtracking.
#     """
#     x = x0
#     s = s_max
#     v = np.multiply(-1, J(x[0], x[1]))
#
#     while np.linalg.norm(np.multiply(s, v), 2) > tolerance:
#
#         # No sense recomputing these every time...
#         fx = F(x[0], x[1])
#
#         jTv = np.dot(J(x[0], x[1]), v)
#
#         xs = x + np.multiply(s, v)
#         while F(xs[0], xs[1]) > fx + delta * s * jTv:
#             s /= 2.0
#             xs = x + np.multiply(s, v)
#
#         x += np.multiply(s, v)
#         v = np.multiply(-1, J(x[0], x[1]))
#
#     return x

def weakest_line(F, J, x0, s_max=0.5, delta=1.0e-03, tolerance=0.5e-08):
    """
    Steepest descent method.
    Weakest line search with backtracking.
    """
    x = x0
    s = s_max
    v = np.multiply(-1, J(x))

    while np.linalg.norm(np.multiply(s, v), 2) > tolerance:

        # No sense recomputing these every time...
        fx = F(x)

        jTv = np.dot(J(x), v)

        xs = x + np.multiply(s, v)
        while F(xs) > fx + delta * s * jTv:
            s /= 2.0
            xs = x + np.multiply(s, v)

        x += np.multiply(s, v)
        v = np.multiply(-1, J(x))

    return x


def steepest_descent_gss(F, J, x0, s_max=0.5, delta=1.0e-03, tolerance=0.5e-08):
    """
    Steepest descent with golden section search.
    
    TODO: actually implement.
    """
    x = x0
    s = s_max
    v = np.multiply(-1, J(x))
    
    while np.linalg.norm(np.multiply(s, v), 2) > tolerance:
        def fs(s):
            xs = x + s * v
            return F(xs)
        
        s = golden_section_search(fs, 0, 1)

        x += np.multiply(s, v)
        v = np.multiply(-1, J(x))
        
    return x

    
def conjugate_gradient(x0, A, b, tolerance=0.5e-08):
    """
    Iterative conjugate gradient method.
    Iterates until the 2 norm of the residual is less than a tolerance.
    """
    x = x0
    d = r = b - A * x

    while np.linalg.norm(r, 2) > tolerance:
        dt = np.transpose(d)
        dtAd = dt * (A * d)
        a = ((dt * r) / dtAd)[0, 0]
        
        x = x + a * d
        
        r = b - A * x
        B = (-1 * (dt * A * r) / dtAd)[0, 0]
        d = r + B * d

    return x

# Nelder-Mead method copied from Numerical Analysis 2nd Edition by
# Timothy Sauer.
# And translated from Matlab to Python by Stephen Bapple
#
# Input: function f, best guess xbar (column vector),
# initial search radius rad and number of steps k
# Output: matrix x whose columns are vertices of simplex,
# function values y of those vertices
def nelder_mead(f, xbar, rad, max_iter=999999, tol=0.5e-6):
    #n = len(xbar);
    xbar = np.array(xbar, dtype=float)
    n = xbar.shape[0]
#    print(n)
    #xbar = np.zeros((n+1, n))
    #xbar[0] = x0
    x = np.zeros((n, n + 1))
    y = np.zeros((n + 1))
#    print(xbar)
#    print('x is ', x)
    x[:, 0] = xbar
    
#    print(x)
#    print(x[:, 0])
    # Done?
    # maybe: 
    # x = np.array((n, n + 1))
    # y 
    x[:,0] = xbar; # each column of x is a simplex vertex
    x[:, 1:n + 1] = xbar * np.ones((1,n)) + rad * np.identity(n)

    for j in range(0, n + 1):
        y[j]=f(x[:,j])  # evaluate obj function f at each vertex

    #[y,r]=sort(y); # sort the function values in ascending order
    r = y.argsort(axis=0)
    y = y[r]

    x = x[:, r] # and rank the vertices the same way

    iter = 1
    while iter < max_iter and np.max(np.ravel(np.abs(x[1:] - x[0]))) > tol\
                          and np.max(np.abs(y[0] - y[1:])) > tol:
        xbar = np.mean[x[:, 0:n].T].T # xbar is the centroid of the face
        xh = x[:, n]          # omitting the worst vertex xh
        xr = 2*xbar - xh 
        yr = f(xr)
        
        if yr < y[n - 1]:
            if yr < y[0]: # try expansion xe
                xe = 3*xbar - 2*xh 
                ye = f(xe)
                
                if ye < yr: # accept expansion
                    x[:, n] = xe
                    y[n] = f(xe)
                else: #% accept reflection
                    x[:,n] = xr 
                    y[n] = f(xr)

            else: # xr is middle of pack, accept reflection
                x[:,n] = xr
                y[n] = f(xr)
        else: # xr is still the worst vertex, contract
            if yr < y[n]: # try outside contraction xoc
                xoc = 1.5*xbar - 0.5*xh
                yoc = f(xoc)
                if yoc < yr: # accept outside contraction
                    x[:,n] = xoc
                    y[n] = f(xoc)
                else: # shrink simplex toward best point
                    for j in range(1, n + 1):
                        x[:,j] = 0.5*x[:,0]+0.5*x[:,j] 
                        y[j] = f(x[:,j])
            
            else: # xr is even worse than the previous worst
                xic = 0.5*xbar+0.5*xh
                yic = f(xic)
                
                if yic < y[n]: # accept inside contraction
                    x[:,n] = xic 
                    y[n] = f(xic)
                else: # shrink simplex toward best point
                    for j in range(1, n + 1):
                        x[:,j] = 0.5*x[:,1]+0.5*x[:,j]
                        y[j] = f(x[:,j])
        
        # Resort the obj function values.
        r = y.argsort(axis=0)
        y = y[r]
        x = x[:, r] # and rank the vertices the same way
        
        iter += 1
        
    return x

################################################################################
# Visualizers                                                                  #
################################################################################

def plot2d_with_mins(f, a, b, mins=()):
    """
    Plot a 2D function.
    """

    x_pts = np.linspace(a, b, 1000)
    fig, ax = plt.subplots()
    ax.plot(x_pts, f(x_pts), color='blue')
    for min in mins:
        ax.scatter(min, f(min), color='red')
    plt.show()


def plot3d_with_mins(F, x_range=(-2, 2), y_range=(-2, 2), mins=()):
    """
    Plot a 3D function.
    """
    
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
            # Rows: y values. Columns: x values
            Z[j, i] = F([x[i], y[j]])
            
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    p1 = ax1.plot_surface(X, Y, Z, cmap=cm.terrain, alpha=(1 if not mins else 0.6))

    # Plot the minimums, if any.
    for min in mins:
        ax1.scatter3D(min[0], min[1], F(min[0], min[1]))

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.show()


################################################################################
# Sentinel. Currently used for testing.                                        #
################################################################################

if __name__ == "__main__":
    def F(x):
        u = x[0]
        v = x[1]
        return 10*u**2 - 16*u*v + 8*v**2 + 8*u - 16*v + 16

    def J(x):
        u = x[0]
        v = x[1]
        return [20*u - 16*v + 8,
                -16*u+16*v - 16]

    def H(x):
        u = x[0]
        v = x[1]
        return [[20, -16],
                [-16, 16]]

    A = np.matrix([[20, -16], [-16, 16]])
    b = np.matrix([[-8],[16]])

    x0 = [[2.5], [2.5]]

    min = conjugate_gradient(x0, A, b)
    print('The minimum is: (%.2f, %.2f, %.2f)' 
          % (min[0], min[1], F(min)))
    
    #plot3d_with_mins(F, [0, 5], [0, 5], mins=[min])
    plot3d_with_mins(F)