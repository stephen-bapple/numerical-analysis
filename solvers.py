'''
Numerical DE Solvers

Author: Stephen Bapple
'''

from numpy import exp, arange
from matplotlib import pyplot


def euler(f, h, y0, t):
    '''
    Implementation of Euler's method.

    Args:
        f: the differential equation.
        h: the step size.
        y0: the initial y value.
        t: the list of t values.

    Returns:
        list: The list of w (approximate y) values.
    '''
    
    # Check if user passed in a callable function.
    # If so, try to extract a list of evals from it.
    if callable(f):
    
    w = [y0]
    for i in range(1, len(t)):
        w.append(w[i - 1] + h * f(t[i - 1], w[i - 1]))

    return w


def back_euler(f, df, ddf, h, y0, t, tolerance=0.0001):
    '''
    Todo: Add doc.
    '''
    def newtons(x0, w):
        x = x0 - (f(x0) * fp(x0)) / (fp(x0)**2 - f(x0) * ddf(x0))
        while abs(x - x0) > tolerance:
            x0 = x
            x = x - (f(x) * fp(x)) / (fp(x)**2 - f(x) * ddf(x))

        return x
    
    w = [y0]
    for i, t in enumerate(t):
        z = newtons(w[i], w[i])
        w.append(w[i], + h * f(t[], z))

        
def trapezoid(f, h, y0, t):
    '''
    Implementation of the Trapezoid Method.

    Args:
        f: the differential equation.
        h: the step size.
        y0: the initial y value.
        t: the list of t values.

    Returns:
        list: The list of w (approximate y) values.
    '''
    w = [y0]
    for i in range(1, len(t)):
        ft = f(t[i - 1], w[i - 1])
        w.append(w[i - 1] + (h / 2) * (ft + f(t[i], w[i - 1] + h * ft)))

    return w

    
def rk4(f, h, y0, t):
    '''
    Runge-Kutta 4 step method.
    Order 4.
    TODO: add docstring.
    '''
    w = [y0]
    for i in range(0, len(t) - 1):
        s1 = f(t[i], w[i])
        s2 = f(t[i] + h / 2, w[i] + (h / 2) * s1)
        s3 = f(t[i] + h / 2, w[i] + (h / 2) * s2)
        s4 = f(t[i] + h, w[i] + h * s3)
        
        w.append(w[i] +  (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4))
        
    return w
    
    
def ab4(f, h, w, t):
    '''
    Adams-Bashforth method.
    This method requires 
    TODO: finish docstring.
    '''
    for i in range(3, len(t) - 1):
        w.append(w[i] + (h / 24) * (55 * f(t[i])\
             - 59 * f(t[i - 1]) + 37 * f(t[i - 2])\
             - 9 * f(t[i - 3])))
    return w

    
def am3():
    '''
    Adams-Moulton method.
    TODO: add docstring
    '''
    pass