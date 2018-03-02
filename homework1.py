'''
Homework 1 Python solutions.
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
    w = [y0]
    for i in range(1, len(t)):
        w.append(w[i - 1] + h * f(t[i - 1], w[i - 1]))

    return w


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


def y1p(t, y):
    '''
    A differential equation for problem 1.
    Args:
        t: The t value to evalutate at.
        y: The y value to evaluate at.

    Returns:
        float: dy/dt at the given y and t.
    '''
    return 4 * t - 2 * y


def y1(t):
    ''' 
    The solution to the DE in problem 1.

    Args:
        t: The t value to evaluate at.
    Returns:
        float: The y value at the given t.
    '''
    return 2 * t + exp(-2 * t) - 1


def y2p(t, y):
    '''
    The differential equation for problem 2.
    Args:
        t: The t value to evalutate at.
        y: The y value to evaluate at.

    Returns:
        float: dy/dt at the given y and t.

    '''
    return 1 - y**2


def y2(t):
    ''' 
    The solution to the DE in problem 2.
    Args:
        t: The t value to evaluate at.
    Returns:
        float: The y value at the given t.

    '''
    return (exp(2 * t) - 1) / (exp(2 * t) + 1)


if __name__ == '__main__':

    ############################################################################
    #   Problem 1 (c)                                                          #
    ############################################################################

    # Obtain the approximate solution via Euler's method.
    t = [0, .25, .5, .75, 1]
    h = 0.25
    y0 = 0
    yp = y1p
    y = y1

    approx_solution = euler(yp, h, y0, t)
    print('The Euler approximated y-coordinates are:\n%r\n' % approx_solution)

    # Obtain the actual solution via the pre-solved solution.
    actual_solution = []
    for t_sub in t: 
        actual_solution.append(y(t_sub))

    print('The actual y-coordinates are:\n%r\n' % actual_solution)

    ############################################################################
    #   Problem 1 (f)                                                          #
    ############################################################################
    trap_approx = trapezoid(yp, h, y0, t)
    print('The Trapezoid approximated y-coordinates are:\n%r\n' % trap_approx)

    ############################################################################
    #   Problem 1 (d) and (g)                                                  #
    ############################################################################

    # Isolate the solution at t = 1
    solution_t1 = actual_solution[-1]

    # Compute the approximate solutions @t=1 for different h values.
    h_k = []
    k_errors_e = []
    k_errors_t = []

    for k in range(0, 6):
        h_k.append(0.1 * 2**(-k))
        t = arange(0, 1 + h_k[-1], h_k[-1])
        approx_k_e = euler(yp, h_k[-1], y0, t)[-1]
        approx_k_t = trapezoid(yp, h_k[-1], y0, t)[-1]
        k_errors_e.append(abs(approx_k_e - solution_t1))
        k_errors_t.append(abs(approx_k_t - solution_t1))

    # Plot the errors.
    _, ax = pyplot.subplots()
    ax.plot(h_k, k_errors_e, label='Global error - Euler')
    ax.plot(h_k, k_errors_t, label='Global error - Trapezoid')
    ax.legend()
    ax.set_xlabel('Step size (h)')
    ax.set_ylabel('Global error at t = 1')
    ax.set_xscale('log')
    ax.set_yscale('log')
    pyplot.show()

    ############################################################################
    #   Problem 2.                                                             #
    ############################################################################
    yp = y2p
    y = y2
    y0 = 0
    h = 0.1
    t = arange(0, 1 + h, h)
    y_vals = y(t)

    #print('t values        : %r' % t)
    #print('true y values   : %r' % y_vals)
    p2_solutions = trapezoid(yp, h, y0, t)
    p2_global_error = []

    #print('approx y values : %r' % p2_solutions)

    # Compute global truncation error.
    for i in range(0, len(y_vals)):
        p2_global_error.append(abs(y_vals[i] - p2_solutions[i]))

    #print('global errors   : %r' % p2_global_error)

    # Compute local truncation error.
    p2_local_error = [0] # Instead of undefined let's just say 0.

    for i in range(0, len(y_vals) - 1):
        #print(t[i:i + 2])
        segment = trapezoid(yp, h, y_vals[i], t[i:i + 2])
        #print('Local segment: %r ' % segment)

        local_error = abs(segment[-1] - y_vals[i + 1])
        p2_local_error.append(local_error)

    #print("%f = %f?" %(p2_global_error[1], p2_local_error[1]))

    # Graph the global and local truncation error.
    fig, ax = pyplot.subplots(2, sharex=True)
    # Approximation vs true solution.
    ax[0].plot(t, p2_solutions, 'b', label='Trapezoid Approximation')
    ax[0].plot(t, y_vals, 'r', label='True Solution') 
    ax[0].legend()
    ax[0].set_xlabel(' t')
    ax[0].set_ylabel(' y ')
    # Global vs. local error.
    ax[1].plot(t, p2_global_error, label='Global Error')
    ax[1].plot(t, p2_local_error, label='Local Error')   
    ax[1].legend()
    ax[1].set_xlabel(' t ')
    ax[1].set_ylabel(' | Forwards Error |')
    pyplot.show()

