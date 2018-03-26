import numpy as np
from matplotlib import pyplot
from matplotlib import animation


def problem():
    ############################################################################
    #   Easy problem                                                           #
    ############################################################################
    #
    # Solving problem Ut = D * Uxx
    # With 
    # f(x, 0) = sin(2 pi x)^2
    #    
    # D = 1
    # Using Forward Temporal spatial

    # Spatial discretization
    h = 0.01
    I = [0, 1]

    # Should be int before conversion, otherwise we have a bad h.
    M = int((I[1] - I[0]) / h)
    print('Number of spatial pts: ', M + 1)
    m = M - 1

    # Discretize x coordinates.
    x = np.arange(0, 1 + h, h)
    x_sub = x[1:-1]

    # print('x values: ', x)
    # print('subset x_values: ', x_sub)
    # print('size(x) == m?', (len(x_sub) == m))
    
    # Diffusivity constant.
    D = 1
    
    # Boundary conditions.
    Uat = 3
    Ubt = 1
    
    # Temporal discretization
    # This value of k satisfies the inequality Dk/h^2 < 1/2 for each interval
    # as well as evenly dividing t = {0.02, 3, 5, 10}
    # k = 1 / 500
    k = 0.000049
    t = np.arange(0, 2, k); print(t)
    print('length of t: ', len(t))
    w = np.matrix(np.zeros((m, len(t))))
    print('shape of w: ', w.shape)
    
    # Theta
    theta = (D * k) / h**2; print('theta is: ', theta)

    # Initial temperature distribution
    def f(_x):
        return np.sin(2 * np.pi * _x)**2

    # Set up theta matrix
    t_mat = np.diagflat( m *[1 - 2* theta])\
      + np.diagflat((m - 1) * [theta], 1)\
      + np.diagflat((m - 1) * [theta], -1)
    t_mat = np.matrix(t_mat)
    print(t_mat)
    
    # Set up the initial temperature distribution.
    # Can do this easily because it is uniform.
    w0 = np.matrix(np.zeros((m, 1)))
    for i in range(0, m):
        w0[i, 0] = f(x_sub[i])
        # w0[i, 0] = f(x[i])
    
    w0[0, 0] = Uat
    w0[-1, 0] = Ubt
    
    print('shape of w0: ', w0.shape)
    w[:, 0] = w0
    print('w after assignment of initial value vector')
    print(w)
    
    s = np.matrix(np.zeros((m, 1)))
    s[0, 0] = Uat * theta
    s[-1, 0] = Ubt * theta
    print(s)
    
    # This part takes a long time.
    for i in range(0, len(t) - 1):
        w[:,i + 1] = ((t_mat * w[:, i]) + s)
   
    print(w)
    
    
    # Do some animation to show the change in temperature over time.
    #
    # The following code segment that does the animation was modified from
    # the matplotlib example at: 
    # https://matplotlib.org/examples/animation/simple_anim.html
    
    fig, ax = pyplot.subplots()
    
    # Somewhat frustrating list manipulation.
    #   Required to include the end points, it seems.
    y_data = w[:, 0].flatten().tolist()[0]
    y_data.append(Ubt)
    y_data.insert(0, Uat)

    time_text = ax.text(0.02, 0.95, 'TIME START', transform=ax.transAxes)
    line, = ax.plot(x, y_data)

    def animate(i):
        y_data = w[:, i].flatten().tolist()[0]
        y_data.append(Ubt)
        y_data.insert(0,Uat)
        
        line.set_ydata(y_data)
        time_text.set_text('time = %.6f seconds' % (i * k))
        return line, time_text

    # Init only required for blitting to give a clean slate.
    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        time_text.set_text('')
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(t) // 10),
                                  init_func=init, interval=5, blit=True)

    ax.set_xlim((-0.2, 1.2))
    ax.set_ylim((-0.2, 3.5))
    ax.set_ylabel('Degrees Fahrenheit')
    ax.set_xlabel('Position in rod.')    
    pyplot.show()


if __name__ == '__main__':
    problem()
