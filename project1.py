import numpy as np
from matplotlib import pyplot as plt
import ode_solvers as solvers

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

# To use when saving figures
fig_number = 1

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
# Not used currently.
def ddf(t, p):
    return -20


# Compare the convergence of a solver with different step sizes against
# the analytical solution.
def compare_convergence(solver, h_set, num_iterations, solution, title='???'):
    global fig_number
    ## Set up array of all solved values.
    w_set = np.zeros((len(h_set), num_iterations))
    
    ## Plot the iterations for each step size.
    fig, ax = plt.subplots(len(h_set) + 1, sharex=True, sharey=True)
    
    for i, h in enumerate(h_set):
        t = [h * i for i in range(num_iterations)]
        w_set[i, :] = solver(f, h, p0, t)
        ax[i].plot(t, w_set[i, :], label='h = %.2f' % h, color='white')
        ax[i].legend(loc='lower right')

    ## Plot true solution. 
    # 500 points randomly chosen for smoothness.
    t = np.linspace(0, num_iterations * h_set[-1], 500)
    ax[-1].plot(t, F(t), label='True solution', color='white')
    
    # Put labels in the correct places.
    ax[0].set_title("Convergence of " + title + " by step size", color='white')
    ax[-1].set_xlabel('time', color='white')
    ax[(len(h_set) + 1) // 2].set_ylabel(' p ', color='white')
    ax[-1].legend(loc='lower right')
    
    # Black background.
    fig.set_facecolor('#000000FF') # Black background
    
    # Change all colors to black/white.
    for a in ax:
        a.set_facecolor('#000000FF') # Black background
        a.xaxis.set_tick_params(color='white', labelcolor='white')
        a.yaxis.set_tick_params(color='white', labelcolor='white')
        for spine in a.spines.values():
            spine.set_color('white')

    # Smoosh all graphs together
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        
    fig.savefig('Figure_%d' % fig_number, transparent=True)
    fig_number += 1
    plt.show()


def bifurcation_plot(solver, h_set, num_iterations=230, depth=30, title='???'):
    global fig_number
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
    
    # Clean us stylistically.
    # Black background.
    ax.set_facecolor('#000000FF') # Black background
    fig.set_facecolor('#000000FF') # Black background
    # White text.
    ax.set_title("Bifurcation diagram for " + title, color='white')
    ax.set_xlabel('h / step size', color='white')
    ax.set_ylabel('approximation of p', color='white')
    ax.xaxis.set_tick_params(color='white', labelcolor='white')
    ax.yaxis.set_tick_params(color='white', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    fig.savefig('Figure_%d' % fig_number, transparent=True)
    fig_number += 1
    plt.show()


### Main script.    
if __name__ == '__main__':
    p0 = 0.1
    granularity = 0.00001

    # Euler's
    print("Euler's method.")
    compare_convergence(solvers.euler, [0.18, 0.23, 0.25, 0.3], 40, F,
                        "Euler's Method")
    bifurcation_plot(solvers.euler, np.arange(0.18, 0.3, granularity),
                     title="Euler's Method")

    # Runge-Kutta 4 stage
    print("Runge-Kutta 4th Order")
    compare_convergence(solvers.rk4, [0.18, 0.23, 0.25, 0.3, 0.325, 0.35, 0.4],
                        60, F, 'Runge-Kutta 4th order')
    bifurcation_plot(solvers.rk4, np.arange(0.18, 0.4, granularity),
                     title='Runge-Kutta 4th order')

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
    print("Adams-Bashforth.")
    bifurcation_plot(solvers.ab4, np.arange(0.01, 0.06940, granularity),
                     title='Adams-Bashforth')

    # Mathematical rearranging of pc4 ab/am
    print("Rearranged pc4.")
    bifurcation_plot(solvers.pco4, np.arange(0.01, 0.1767, granularity),
                     title='rearranged predictor/corrector 4th order')

    # Second order predictor/corrector
    print("2nd order predictor/corrector.")
    bifurcation_plot(solvers.pc2, np.arange(0.01, 0.02, granularity),
                     title='2nd order predictor/corrector')

    # Trapezoid
    print("Trapezoid.")
    bifurcation_plot(solvers.trapezoid, np.arange(0.18, 0.34037, granularity),
                     title='trapezoid method')
