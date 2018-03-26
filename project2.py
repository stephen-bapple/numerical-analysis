"""
Project #2 for Numerical Analysis II

This project revolves around applying various optimization techniques to the
following system: TBD

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


def u(points):
    x = [i for i in points[::3]]
    y = [i for i in points[1::3]]
    z = [i for i in points[2::3]]

    potential = 0

    for j in range(1, len(x)):
        for i in range(0, j):
            #r = x[i]*x[i] + y[i]*y[i] + z[i]*z[i]\
            #    + x[j]*x[j] + y[j]*y[j] + z[j]*z[j]\
            #    - 2*(x[i]*x[j] + y[i]*y[j] + z[i]*z[j])
            #r = r**(1/2)

            if i == j:
                print('i should never equal j')

            r = ((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)#**(1/2)
            #print('ij: %d|%d' % (i, j))
            #print('r: %d | for: (%d, %d, %d) vs (%d, %d, %d)'
            #      % (r, x[i], y[i], z[i], x[j], y[j], z[j]))
            potential += r**-6 - 2*r**-3

    #print('potential:', potential)
    return potential


def main():
    mode = 2

    # Number of points
    n = 13

    # First two points, which are trivial.
    x = [0, 0]
    y = [0, 0]
    z = [0, 1]
    solution = np.array([0, 0, 0,
                         0, 0, 1])
    min_solution = None
    fig = plt.figure()

    # Find every structure from 3 up to n
    for i in range(3, n + 1):
        if i >= 13:
            num_randoms = 200
        else:
            num_randoms = 10

        if mode == 1:
            while u(solution) > -44.326801:
                intermediate = np.append(solution, [uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)])

                v = minimize(u, intermediate, method='Powell', tol=0.5e-10)
                print(u(v.x), end=', ')
                if min_solution is None or u(min_solution) > u(v.x):
                    min_solution = v.x

            print('.')
            solution = min_solution
        elif mode == 2:
            for le in range(num_randoms):
                intermediate = np.append(solution, [uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)])

                v = minimize(u, intermediate, method='Powell', tol=0.5e-10)
                print(u(v.x), end=', ')
                if min_solution is None or u(min_solution) > u(v.x):
                    min_solution = v.x

            print('.')
            solution = min_solution

        elif mode == 3:
            mag = .5
            perturbations = [[0, 0, mag], [0, 0, -mag], [0, mag, 0], [0, mag, 0], [mag, 0, 0], [mag, 0, 0]]

            for pert in perturbations:
                for p in range(0, len(solution), 3):
                    new_x = solution[i] + pert[0]
                    new_y = solution[i + 1] + pert[1]
                    new_z = solution[i + 2] + pert[2]

                    intermediate = np.append(solution, [new_x, new_y, new_z])

                    v = minimize(u, intermediate, method='Powell', tol=0.5e-10)
                    print(u(v.x), end=', ')
                    if min_solution is None or u(min_solution) > u(v.x):
                        min_solution = v.x

            print('.\n::::%d: %10f ::::' % (i, u(min_solution)))
            solution = min_solution

    x = [x for x in solution[::3]]
    y = [y for y in solution[1::3]]
    z = [z for z in solution[2::3]]

    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(x, y, z, c='r', s=100)

    for j in range(1, len(x)):
        for k in range(0, j):
            ax1.plot([x[k], x[j]], [y[k], y[j]], [z[k], z[j]], color='k', alpha=0.3)

    print(solution)
    print('Energy: %10f' % u(solution))

    ax1.set_xlim3d(-2, 2)
    ax1.set_ylim3d(-2, 2)
    ax1.set_zlim3d(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    fig.suptitle('n = %d' % i)
    ax1.legend()
    plt.show()


if __name__ == '__main__':
    main()
