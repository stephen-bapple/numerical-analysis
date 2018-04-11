"""
Project #2 for Numerical Analysis II

This project revolves around applying various optimization techniques
to optimize Lennard-Jones clusters.

Optimization techniques used are found in optimizers.py

Author: Stephen Bapple
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import optimizers as opt
from scipy.optimize import minimize
from random import uniform

# Global minimums for Lennard-Jones clusters 3 through 13.
# Source: http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html
# Last accessed 2018-04-09
               #              3          4           5          6
true_energies = [0, 0, 0, -3.000000, -6.000000, -9.103852, -12.712062,

                 #    7              8        9           10          11
                 -16.505384, -19.821489, -24.113360, -28.422532, -32.765970,

                 #    12            13        14          15          16
                 -37.967600, -44.326801, -47.845157, -52.322627, -56.815742,

                 #    17            18        19          20          21
                 -61.317995, -66.530949, -72.659782, -77.177043, -81.684571,

                 #    22            23        24          25          26
                 -86.809782, -92.844472, -97.348815, -102.372663, -108.315616,

                 #    27            28        29            30 
                 -112.873584, -117.822402, -123.587371, -128.286571, 

                 #    31            32       
                 -133.586422, -139.635524]


def u(points):
    """
    Potential function.
    """
    potential = 0

    # The potential is the sum of all the forces between each pair of molecules.
    for j in range(3, len(points), 3):
        for i in range(0, j, 3):
            r = ((points[i] - points[j])**2 + (points[i + 1] - points[j + 1])**2
                 + (points[i + 2] - points[j + 2])**2)
            potential += r**-6 - 2*r**-3

    return potential


def du(points):
    """
    Jacobian for the potential function.
    """

    v = [0] * len(points)
    #for k in range(len(points) // 3):
    for k in range(len(points)):
        potential = 0

        for j in range(3, len(points), 3):
            for i in range(0, j, 3):
                # Check if we're taking the derivative relative to this variable.
                if i <= k <= i + 2:
                    if k == i:
                        jk = j
                    elif k == i + 1:
                        jk = j + 1
                    else:  # k == i + 2
                        jk = j + 2

                    r = ((points[i] - points[j])**2
                         + (points[i + 1] - points[j + 1])**2
                         + (points[i + 2] - points[j + 2])**2)

                    potential += 12*(-r**-7 + r**-4)*(points[k] - points[jk])

                else:
                    r = ((points[i] - points[j])**2
                         + (points[i + 1] - points[j + 1])**2
                         + (points[i + 2] - points[j + 2])**2)
                    potential += r ** -6 - 2 * r ** -3

        v[k] = potential

    return np.array(v)


def plot_structure(points):
    x = [x for x in points[::3]]
    y = [y for y in points[1::3]]
    z = [z for z in points[2::3]]

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(x, y, z, c='r', s=100)

    for j in range(1, len(x)):
        for k in range(0, j):
            ax1.plot([x[k], x[j]], [y[k], y[j]], [z[k], z[j]], color='k', alpha=0.3)

    ax1.set_xlim3d(-2, 2)
    ax1.set_ylim3d(-2, 2)
    ax1.set_zlim3d(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    fig.suptitle('n = %d' % (len(points) // 3))
    plt.show()


def animate_structure():
    pass


def main():
    n = 7
    errors = try_steepest_descent(n)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(3, n + 1)], errors)
    ax.set_yscale('log')
    plt.show()
    #try_nelder_mead()


def try_steepest_descent(n):
    errors = []
    
    # First two points, which are trivial.
    initial_solution = np.array([0.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0])
    #\________________________________________________________________________/#
    #                                                                          #
    #   Section 1:                                                             #
    #   Attempt to use steepest descent with weakest line search               #
    #\________________________________________________________________________/#

    solution = initial_solution
    min_solution = None
    umin = 0
    for i in range(3, n + 1):
        #num_randoms = 1000
        num_randoms = 10000
        for j in range(num_randoms):
            intermediate = np.append(solution, [uniform(-1, 1),
                                                uniform(-1, 1),
                                                uniform(-1, 1)])
            while u(intermediate) > -1:
                intermediate = np.append(solution, [uniform(-1, 1),
                                                    uniform(-1, 1),
                                                    uniform(-1, 1)])
                #print('Stuck in finding a good initial guess', flush=True)

            v = opt.weakest_line(u, du, intermediate, tol=0.5e-10)

            #print(u(v), end=', ', flush=True)
            uv = u(v)
            
            if umin > uv:
                min_solution = v
                umin = uv
            print('Done: %.2f%%' % ((j / num_randoms) * 100), end='\r')
        #print('-------------------------')
        #print('Number of atoms: %d\nMinimum Energy: %10f\n' % (i, u(min_solution)))
        print('--------------------------')
        print('Number of atoms: %d\n' % i + 
              'Minimum Energy: %10f\n' % umin)
        solution = min_solution
        errors.append(abs(true_energies[i] - umin))
    print(solution)
    print('Energy: %10f' % u(solution))

    plot_structure(solution)

    return errors


def try_nelder_mead():
    errors = []

    # First two points, which are trivial.
    initial_solution = np.array([0, 0, 0,
                                 0, 0, 1])

    #\________________________________________________________________________/#
    #                                                                          #
    # Section 2:                                                               #
    # Use Nelder-Mead                                                          #
    #\________________________________________________________________________/#

    # Use 7 points.
    n = 4
    solution = initial_solution
    min_solution = None

    # Find every structure from 3 up to n
    for i in range(3, n + 1):
        num_randoms = 10

        for _ in range(num_randoms):
            intermediate = np.append(solution, [uniform(-1, 1), uniform(-1, 1),
                                                uniform(-1, 1)])
            while u(intermediate) > 10:
                intermediate = np.append(solution, [uniform(-1, 1),
                                                    uniform(-1, 1),
                                                    uniform(-1, 1)])
                print('stuck priming', flush=True)
            v = opt.nelder_mead(u, intermediate, .5, max_iter=999, xtol=0.5e-6, ftol=0.5e-6)
            v = v[0][:-1]
            # print(u(v), end=', ')
            if min_solution is None or u(min_solution) > u(v):
                min_solution = v
            print('stuck calculating', flush=True)
        print('::::%d: %10f ::::' % (i, u(min_solution)))
        solution = min_solution

    print(solution)
    print('Energy: %10f' % u(solution))

    plot_structure(solution)


if __name__ == '__main__':
    main()
