import numpy as np
from optimizers import golden_section_search, multivariate_newtons,\
                       weakest_line,\
                       steepest_descent_gss, conjugate_gradient,\
                       plot2d_with_mins, plot3d_with_mins,\
                       conjugate_gradient_search

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
#demo_weakest_line()


def demo_weakest_line_gss():
    def F(x, y):
        return x**4 + y**4 + 2 * x**2 * y**2 + 6 * x * y - 4 * x - 4 * y + 1

    # Jacobian
    def J(x, y):
        return [4 * x**3 + 4 * x * y**2 + 6 * y - 4, 
                4 * y**3 + 4 * x**2 * y + 6 * x - 4]
    def H(x, y):
        return [[12*x**2 + 4 * y**2, 8*x*y + 6],
                [8 * x * y + 6, 12*y**2 + 4*x**2]]

    x0 = [-1, 1]
    min = []
    min.append(steepest_descent_gss(F, J, x0))
    #min.append(weakest_line(F, J, x0))
    #min.append(multivariate_newtons(J,H,x0))
    plot3d_with_mins(F, mins=min)
    print(min)
#demo_weakest_line_gss()


def demo_conjugate_gradient_search():
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

    #x0 = [[2.5], [2.5]]
    x0 = [2.5, 2.5]
    min = conjugate_gradient_search(F, J, x0)
    print('The minimum is: (%.2f, %.2f, %.2f)' 
          % (min[0], min[1], F([min[0], min[1]])))
    
    plot3d_with_mins(F, [0, 5], [0, 5], mins=[min])

#demo_conjugate_gradient_search()

if __name__ == "__main__":
    demo_conjugate_gradient_search()