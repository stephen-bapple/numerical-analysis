import numpy as np
from matplotlib import pyplot as plt
import solvers

################################################################################
# This project deals with the following differential equation:
#
# dp/dt = 10p(1 - p)
#
# Part a: Solution:
#
# p(t) = (e^10t) / (e^10t + 9)
#
# It should be clear from the form of p(t) that lim p(t) -> 1 as t-> infinity.
#
################################################################################

# Global constants
p0 = 0.1

# Define the function
def f(t, p):
    return 10 * p * (1 - p)


# Define the analytical solution
def F(t):
    return np.exp(10 * t) / (np.exp(10 * t) + 9)


# Derivative of the function
def df(t, p):
    return 10 - 20 * p


# Second derivative of the function
def ddf(t, p):
    return -20


# Compare the convergence of a solver with different step sizes against
# the analytical solution.
def compare_convergence(solver, h_set, num_iterations, solution, title='???'):
    ## Set up array of all solved values.
    w_set = np.zeros((len(h_set), num_iterations))
    
    ## Plot the iterations for each step size.
    fig, ax = plt.subplots(len(h_set) + 1, sharex=True, sharey=True)
    for i, h in enumerate(h_set):
        t = [h * i for i in range(num_iterations)]
        w_set[i, :] = solver(f, h, p0, t)
        ax[i].plot(t, w_set[i, :], label='h = %.2f' % h)
        ax[i].legend(loc='lower right')

    ## Plot true solution. 
    # 500 points randomly chosen for smoothness.
    t = np.linspace(0, num_iterations * h_set[-1], 500)
    ax[-1].plot(t, F(t), label='True solution')
    
    # Put labels in the correct places.
    ax[0].set_title("Convergence of " + title + " by step size")
    ax[-1].set_xlabel('time')
    ax[(len(h_set) + 1) // 2].set_ylabel(' p ')
    ax[-1].legend(loc='lower right')
    
    # Smoosh all graphs together
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    plt.show()
    
def bifurcation_plot(solver, h_set, num_iterations=230, depth=30, title='???'):
    # Prepare the 2d array of solved points.
    w_set = np.zeros((len(h_set), depth))

    # Run method to obtain convergent w value for each h value,
    # using variable 'depth' as the number of results to show any oscillation. 
    for i, h in enumerate(h_set):
        t = [h * j for j in range(num_iterations + 1)]
        w_set[i, :] = solver(f, h, p0, t)[num_iterations + 1 - depth:]

    # Plot h vs y
    fig, ax = plt.subplots()
    for i in range(depth):
        ax.plot(h_set, w_set[:, i], color='white', ls='', marker=',')
    ax.set_facecolor('#000000FF') # Black background
    #fig.set_facecolor('#000000FF')
    ax.set_title("Bifurcation diagram for " + title)
    ax.set_xlabel('h / step size')
    ax.set_ylabel('approximation of y')
    plt.show()


### Plot the bifurcation diagram to show devolution into chaos.
def srk2():
    #h_set = np.linspace(0.18, 0.3, 10000)
    width = 30
    #h_set = np.arange(0.0001, 0.17720, 0.00001)
    #h_set = np.arange(0.0001, 0.1767, 0.00001)
    h_set = np.arange(0.0001, 0.4, 0.00001)
    w_set = np.zeros((len(h_set), width))

    # Run method to obtain y value method should converge to for each h value
    # using last 30 results to show any oscillation. 
    for i, h in enumerate(h_set):
        t = [h * j for j in range(231)]
        w_set[i, :] = rk2(f, h, p0, t)[231 - width:]

    # Plot h vs y
    fig, ax = plt.subplots()
    colors = ['white', 'red', 'blue']
    for i in range(width):
        ax.plot(h_set, w_set[:, i], color=colors[0], ls='', marker=',')
    ax.set_facecolor('#000000FF')
    #fig.set_facecolor('#000000FF')
    plt.show()


### Main script.    
if __name__ == '__main__':
    granularity = 0.0001
    # Euler's
    
    print("Euler's method.")
    ##compare_convergence(solvers.euler, [0.18, 0.23, 0.25, 0.3], 40, F,
    ##                    "Euler's Method")
    bifurcation_plot(solvers.euler, np.arange(0.18, 0.3, granularity),
                     title="Euler's Method")
    
    # Runge-Kutta 4 stage
    print("Runge-Kutta 4th Order")
    ##compare_convergence(solvers.rk4, [0.18, 0.23, 0.25, 0.3, 0.325, 0.35, 0.4],
    ##                    60, F, 'Runge-Kutta 4th order')
    bifurcation_plot(solvers.rk4, np.arange(0.18, 0.4, granularity),
                     title='Runge-Kutta 4th order')
    '''
    # Backwards Eulers
        # Note: this fails because Newton's method does not converge.
    #compare_convergence(back_euler, [0.18, 0.23, 0.25, 0.3], 40, F,
    #                    "Backwards Euler's Method")

    # Predictor/corrector Adams-Bashforth / Adams-Moulton Order 4
    # Uses rk4.
    print("Adams-Bashforth/Adams-Moulton Predictor/Corrector.")
    bifurcation_plot(solvers.predictor_corrector4,
                     np.arange(0.01, 0.1767, granularity),
                     title='predictor/corrector 4th order')

    # Adams Bashforth
    # 90 might be unnecessary... CHECK?
    print("Adams-Bashforth.")
    bifurcation_plot(solvers.ab4, np.arange(0.01, 0.06940, granularity),
                     231, 90, title='Adams-Bashforth')

    # Mathematical rearranging of pc4 ab/am
    print("Rearranged pc4.")
    bifurcation_plot(solvers.pco4, np.arange(0.01, 0.1767, granularity),
                     title='rearranged predictor/corrector 4th order')

    # Second order predictor/corrector
    print("2nd order predictor/corrector.")
    bifurcation_plot(solvers.pc2, np.arange(0.01, 0.02, granularity),
                     title='2nd order predictor/corrector')
    '''
    # Trapezoid
    print("Trapezoid.")
    bifurcation_plot(solvers.trapezoid, np.arange(0.18, 0.3404, granularity),
                     title='trapezoid method')
