'''
Numerical Differential Equation Solvers

Author: Stephen Bapple

TODO: finish docstrings for all methods.
'''

from numpy import exp, arange, zeros
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
    
    w = [y0]
    for i in range(1, len(t)):
        w.append(w[i - 1] + h * f(t[i - 1], w[i - 1]))

    return w


def back_euler(f, df, ddf, h, y0, t, tolerance=0.01):
    '''
    Implementation of backwards Euler's method.
    
    Args:
        f: The differential equation.
        df: The derivative of the differential equation.
        ddf: The second derivatie of the differential equation.
        h: The step size.
        y0: The initial y value.
        t: The list of t values.
        tolerance: The optional tolerance to use for Newton's method.
        
    Returns:
        list: The list of w (approximate y) values.
    '''
    def newtons(x0, w, h, tol):
        def g(x, w, h):
            return x - (w + h * f(0, x))
        def gp(x, h):
            return 1 - h * df(0, x)
            
        x = x0 - g(x0,w,h) / gp(x0,h)
        print(x-x0)
        while abs(x-x0) > tol:
            print(abs(x-x0))
            x0=x
            x = x0 - g(x0, w, h) / gp(x0, h)
        return x

    w = [y0]
    for i in range(len(t) - 1):
        z = newtons(w[i], w[i], h, tolerance)
        w.append(w[i] + h * f(t[i + 1], z))
    
    return w


def trapezoid_explicit(f, h, y0, t):
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
    
    
def ab4(f, h, y0, t):
    '''
    Adams-Bashforth method.
    This method requires 
    TODO: finish docstring.
    '''
    w = zeros((len(t)))
    w[0:4] = rk4(f, h, y0, t[0:4])
    for i in range(3, len(t) - 1):
        w[i + 1] = w[i] + (h / 24) * (55 * f(t[i], w[i])\
             - 59 * f(t[i - 1], w[i - 1]) + 37 * f(t[i - 2], w[i - 2])\
             - 9 * f(t[i - 3], w[i - 3]))
    #print(w)
    return w

    
def predictor_corrector4(f, h, y0, t):
    ## Step methods to use later.
    def am3_step(f, t, w, h):
        return w[2] + (h / 24) * (9 * f(t[3], w[3])\
                    + 19 * f(t[2], w[2])\
                    - 5 * f(t[1], w[1])\
                    + f(t[0], w[0]))

    def ab4_step(f, t, w, h):
        return w[3] + (h / 24) * (55 * f(t[3], w[3])\
                    - 59 * f(t[2], w[2])\
                    + 37 * f(t[1], w[1])\
                    - 9 * f(t[0], w[0]))
    w = zeros((len(t)))
    
    ## Use runge-kutta to obtain initial values.
    w[0:4] = rk4(f, h, y0, t[0:4])

    for i in range(3, len(t) - 1):
        w[i + 1] = ab4_step(f, t[i - 3:i + 1], w[i - 3:i + 1], h) # predictor
        w[i + 1] = am3_step(f, t[i - 2:i + 2], w[i - 2:i + 2], h) # corrector

    return w


def pco4(f, h, y0, t):
    ## Step methods to use later.
    def am3_step(f, t, w, h):
        return w[2] + (9 * f(t[3], w[3])\
                    + 19 * f(t[2], w[2])\
                    - 5 * f(t[1], w[1])\
                    + f(t[0], w[0])) * (h / 24)

    def ab4_step(f, t, w, h):
        return w[3] + (55 * f(t[3], w[3])\
                    - 59 * f(t[2], w[2])\
                    + 37 * f(t[1], w[1])\
                    - 9 * f(t[0], w[0])) * (h/ 24)
    w = zeros((len(t)))
    
    ## Use runge-kutta to obtain initial values.
    w[0:4] = rk4(f, h, y0, t[0:4])
    
    for i in range(3, len(t) - 1):
        w[i + 1] = ab4_step(f, t[i - 3:i + 1], w[i - 3:i + 1], h) # predictor
        w[i + 1] = am3_step(f, t[i - 2:i + 2], w[i - 2:i + 2], h) # corrector
        
    return w
    
def pc2(f, h, y0, t):
    ## Step methods to use later.
    def ab2_step(f, t, w, h):
        return w[1] + (h / 2) * (3 * f(t[1], w[1]) - f(t[0], w[0]))

    def am2_step(f, t, w, h):
        return w[0] + (h / 2) * (f(t[1], w[1]) - f(t[0], w[0]))
        
        
    w = zeros((len(t)))
    
    # 2 stage midpoint method.
    def mp2(f, h, y0, t):
        w = [y0]
        for i in range(0, len(t) - 1):
            s1 = w[i] + (h / 2) * f(t[i], w[i])
            s2 = f(t[i] + (h / 2), s1)
            w.append(w[i] +  h * s2)
        
        return w
    ## Use midpoint method to obtain initial values.
    w[0:2] = mp2(f, h, y0, t[0:2])
    
    for i in range(2, len(t) - 1):
        w[i + 1] = ab2_step(f, t[i - 1:i + 1], w[i - 1:i + 1], h) # predictor
        w[i + 1] = am2_step(f, t[i:i + 2], w[i:i + 2], h) # corrector

    return w



def rk2(f, h, y0, t):
    w = [y0]
    for i in range(0, len(t) - 1):
        s1 = w[i] + (h / 2) * f(t[i], w[i])
        s2 = f(t[i] + (h / 2), s1)
        w.append(w[i] +  h * s2)
        
    return w


def trapezoid(f, h, y0, t):
    '''
    Implicit trapezoid. Simple predictor/corrector.
    '''
    w = [y0]
    for i in range(0, len(t) - 1):
        w.append(w[i] + h * f(t[i], w[i]))
        w[i + 1] = w[i] + (h / 2) * (f(t[i], w[i]) + f(t[i + 1], w[i + 1]))
        
    return w
