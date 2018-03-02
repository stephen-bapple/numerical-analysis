'''
Homework 2 Python solutions

Some code modified from the examples provided at:
http://sites.msudenver.edu/hbouwmee/mth-4490/
'''

import numpy as np
from matplotlib import pyplot
from matplotlib import animation

def uf(t, u):
    return np.array([u[1], np.exp(t) + u[0] * np.cos(t) - (t + 1) * u[1]])


def vf(t, v):
    return np.array([v[1], v[0] * np.cos(t) - (t + 1) * v[1]])


def rk4_step(f, h, w, t):
    s1 = f(t, w)
    s2 = f(t + h / 2, w + (h / 2) * s1)
    s3 = f(t + h / 2, w + (h / 2) * s2)
    s4 = f(t + h, w + h * s3)

    return w + (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4)


def problem1():
    ############################################################################
    #   Problem 1                                                              #
    ############################################################################

    # Set up boundary values and constants.
    I = [0, 1]
    ya = 1
    yb = 3
    h = 0.01

    n = int((I[1] - I[0]) / h)

    u = np.zeros((n + 1, 2))
    v = np.zeros((n + 1, 2))
    w = np.zeros((n + 1))

    t = [I[0] + i * h for i in range(n + 1)]

    u[0,:] = [ya, 0]
    v[0,:] = [0, 1]

    # Obtain intermediate solutions for the overshoot and undershoot.
    for i in range(n):
        u[i + 1, :] = rk4_step(uf, h, u[i, :], t[i])
        v[i + 1, :] = rk4_step(vf, h, v[i, :], t[i])
        
    # Obtain approximations from the overshoot and undershoot.
    for i in range(n + 1):
        w[i] = u[i, 0] + ((yb - u[n, 0]) / v[n, 0]) * v[i, 0]

    # Print table of values.
    # Since the true solution is somewhat difficult, we will forgo the error.
    print('\n t   |      u     |      v     |      w     ')
    print('-----+------------+------------+-------------')
    for i in range(n + 1):
        print('%1.2f | %1.8f | %1.8f | %1.8f' %
                (t[i], u[i, 0], v[i, 0], w[i]))
    print('\n')

    # Plot the overshoot, undershoot, and the approximation.
    _,ax = pyplot.subplots()
    ax.plot(t, u[:,0], label='IVP_1')
    ax.plot(t, v[:,0], label='IVP_2')
    ax.plot(t, w, label='Approximation')
    ax.legend()
    ax.set_xlim(I)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    pyplot.show()
    

def problem2():
    ############################################################################
    #   Problem 2                                                              #
    ############################################################################
    #
    # Solving problem Ut = D * Uxx
    # With 
    #
    # D = .34 on x in [0, 4]        | (Glass pane 1)
    #      22 on x in [4, 18]       | (Argon gas)
    #     .34 on x in [18, 22]      | (Glass pane 2)
    #
    # Using Forward Temporal spatial
    #
    # Spatial map:
    #
    #  0        4                          18       22
    # -|------- |---------------------------|--------|
    #  | window |       Argon gas           | window |
    #
    #  a        b                           c        d 
    # TODO:
    #   - Break problem into the three intervals.
    #   - Refactor the organization of the initial conditions. 
    # 
    
    # Spatial discretization
    h = 0.02
    I = [0, 22]
    # Should be int automatically, otherwise we have a problem with h and interval.
    M = int((I[1]  - I[0])/ h); print('Number of spatial pts: ', M + 1)
    m = M - 1
    
    # Diffusivity constants
    D = [0.34, 22, 0.34]
    
    # Temporal discretization
    # This value of k satisfies the inequality Dk/h^2 < 1/2 for each interval
    # as well as evenly dividing t = {0.02, 3, 5, 10}
    k = 1 / 110050
    #k = 0.000005
    t = np.arange(0, 2, k); print(t)
    print('length of t: ', len(t))
    w = np.matrix(np.zeros((m, len(t))))
    print('shape of w: ', w.shape)
    
    
    # Boundary indices for the three sections
    a = 0
    b = int(4 / k)
    c = int(18 / k)
    d = int(22 / k)
    print('0 ------ 4 ----------- 18 ------- 22')
    print('%r ------ %r ----------- %r ------- %r' % (a, b, c, d))
    
    # Theta
    theta = [(d * k) / h**2 for d in D]; print('theta is: ', theta)

    # Initial temperature distribution
    Ux = 6
    
    # Set up theta matrix
    t_mat = np.diagflat( m *[1 - 2* theta[0]])\
      + np.diagflat((m - 1) * [theta[0]], 1)\
      + np.diagflat((m - 1) * [theta[0]], -1)
    t_mat = np.matrix(t_mat)
    print(t_mat)
    
    # Set up the initial temperature distribution.
    # Can do this easily because it is uniform.
    w0 = np.matrix(np.ones((m, 1)) * Ux)
    
    print('shape of w0: ', w0.shape)
    w[:, 0] = w0
    print('w after assignment of initial value vector')
    print(w)
    
    s = np.matrix(np.zeros((m, 1)))
    s[0, 0] = 70 * theta[0]
    s[-1, 0] = Ux * theta[0]
    print(s)
    
    # This part takes a long time.
    for i in range(0, len(t) - 1):
        w[:,i + 1] = (t_mat * w[:, i]) + s
   
    print(w)
    
    
    # Do some animation to show the change in temperature over time.
    #
    # The following code segment that does the animation was modified from
    # the matplotlib example at: 
    # https://matplotlib.org/examples/animation/simple_anim.html
    x = np.arange(0, 22 + h, h)
    fig, ax = pyplot.subplots()
    ax.set_ylim((0, 80))
    
    # Somewhat frustrating list manipulation.
    #   Required to include the end points, it seems.    
    y_data = w[:, 0].flatten().tolist()[0]
    y_data.append(6)
    y_data.insert(0, 70)
    line, = ax.plot(x, y_data)

    def animate(i):
        y_data = w[:, i].flatten().tolist()[0]
        y_data.append(6)
        y_data.insert(0, 70)
        line.set_ydata(y_data)
        return line,


    # Init only required for blitting to give a clean slate.
    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(0, 200000, 100),
                                  init_func=init, interval=5, blit=True)
    pyplot.show()

if __name__ == '__main__':
    #problem1()
    problem2()