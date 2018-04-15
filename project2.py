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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm
import optimizers as opt
from scipy.optimize import minimize
from random import uniform
from lennard_jones_functions import u, du

# Global minimums for Lennard-Jones clusters 3 through 32.
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

     
def plot_structure(points, energy, error):
    x = [x for x in points[::3]]
    y = [y for y in points[1::3]]
    z = [z for z in points[2::3]]

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(x, y, z, c='b', s=100)

    for j in range(1, len(x)):
        for k in range(0, j):
            ax1.plot([x[k], x[j]], [y[k], y[j]], [z[k], z[j]], color='k', alpha=0.2)

    ax1.set_xlim3d(-2, 2)
    ax1.set_ylim3d(-2, 2)
    ax1.set_zlim3d(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    fig.suptitle('n = %d\n' % (len(points) // 3) +
                 'Total Absolute Forward Error = %.5f' % error)
    plt.show()


def try_optimizer(n, initial_solution, optimizer, u, num_randoms, **kwargs):
    errors = []
    progress = 0
    solution = initial_solution
    min_solution = None

    for i in range(3, n + 1):
        progress = 0
        a = 0.8
        umin = 0

        # Try num_randoms many guesses.
        for _ in range(num_randoms):
            intermediate = np.append(solution, [uniform(-a, a),
                                                uniform(-a, a),
                                                uniform(-a, a)])

            while u(intermediate) > -1:
                intermediate = np.append(solution, [uniform(-a, a),
                                                    uniform(-a, a),
                                                    uniform(-a, a)])

            # TODO: Find a cleaner way to do this. 
            # 
            # Decide if we are using a jacobian or derivative free method.
            #print('starting optimizer')
            if 'J' in kwargs:
                v = optimizer(F=u, x0=intermediate, **kwargs)
            else:
                v = optimizer(u, intermediate, **kwargs)

            # Update minimum solution if necessary.
            uv = u(v)
            if umin > uv:
                min_solution = v
                umin = uv
                
            # Update progress bar.
            progress += 1
            print('Done: %.2f%%' % ((progress / num_randoms) * 100), end='\r')
            
        print('--------------------------')
        print('Number of atoms: %d\n' % i + 
              'Estimated Minimum Energy : %10f\n' % umin + 
              'Actual Minimum Energy    : %10f\n' % true_energies[i] + 
              'Absolute error           : %10f\n' % abs(true_energies[i] - umin))

        solution = min_solution
        errors.append(abs(true_energies[i] - umin))
        
        # Translate structure back to origin.
        translate_to_origin(solution)

    print(solution)
    print('Energy: %10f' % umin)

    plot_structure(solution, umin, errors[-1])

    return errors


def try_all_random(n, initial_solution, optimizer, u, num_randoms, **kwargs):
    errors = []
    progress = 0
    solution = initial_solution
    min_solution = None

    for i in range(3, n + 1):
        progress = 0
        a = 0.8
        umin = 0

        # Try num_randoms many guesses.
        for _ in range(num_randoms):
            intermediate = []
            for _ in range(len(solution) + 3):
                intermediate.append(uniform(-a, a))

            while u(intermediate) > -1:
                for _ in range(len(solution) + 3):
                    intermediate.append(uniform(-a, a))
            
            # TODO: Find a cleaner way to do this. 
            # 
            # Decide if we are using a jacobian or derivative free method.
            #print('starting optimizer')
            if 'J' in kwargs:
                v = optimizer(F=u, x0=intermediate, **kwargs)
            else:
                v = optimizer(u, intermediate, **kwargs)
            #print('Optimizer finished')
            # Update minimum solution if necessary.
            uv = u(v)
            if umin > uv:
                min_solution = v
                umin = uv

            # Update progress bar.
            progress += 1
            print('Done: %.2f%%' % ((progress / num_randoms) * 100), end='\r')
            
        print('--------------------------')
        print('Number of atoms: %d\n' % i + 
              'Estimated Minimum Energy : %10f\n' % umin + 
              'Actual Minimum Energy    : %10f\n' % true_energies[i] + 
              'Absolute error           : %10f\n' % abs(true_energies[i] - umin))
        
        solution = min_solution
        errors.append(abs(true_energies[i] - umin))
        
        # Translate structure back to origin.
        translate_to_origin(solution)
        
    print(solution)
    print('Energy: %10f' % umin)

    plot_structure(solution, umin, errors[-1])

    return errors


def compare_tolerances(n, initial_solution, u, du, num_randoms, tol1, tol2):
    errors1 = []
    errors2 = []
    progress = 0
    solution1 = initial_solution
    solution2 = initial_solution
    min_solution1 = None
    min_solution2 = None

    for i in range(3, n + 1):
        progress = 0
        a = 0.8
        umin1 = 0
        umin2 = 0

        # Try num_randoms many guesses.
        for _ in range(num_randoms):
            random_pt = [uniform(-a, a), uniform(-a, a), uniform(-a, a)]
            intermediate1 = np.append(solution1, random_pt)
            intermediate2 = np.append(solution2, random_pt)
            
            while u(intermediate1) > -1:
                random_pt = [uniform(-a, a), uniform(-a, a), uniform(-a, a)]
                intermediate1 = np.append(solution1, random_pt)
                intermediate2 = np.append(solution2, random_pt)
            
            v1 = opt.weakest_line(u, du, intermediate1, tol=tol1)
            v2 = opt.weakest_line(u, du, intermediate2, tol=tol2)
            
            # Update minimum solution if necessary.
            uv1 = u(v1)
            if umin1 > uv1:
                min_solution1 = v1
                umin1 = uv1
            
            # Update minimum solution if necessary.
            uv2 = u(v2)
            if umin2 > uv2:
                min_solution2 = v2
                umin2 = uv2
            
            # Update progress bar.
            progress += 1
            print('Done: %.2f%%' % ((progress / num_randoms) * 100), end='\r')
            
        print('--------------------------')
        print('Number of atoms: %d\n' % i + 
              'Tolerance:                 %.4e | %.4e\n' %(tol1, tol2) +
              'Estimated Minimum Energy : %10f | %10f\n' % (umin1, umin2) +  
              'Absolute error           : %10f | %10f\n'
              % (abs(true_energies[i] - umin1), abs(true_energies[i] - umin2)) +
              'Actual Minimum Energy    : %10f\n' % true_energies[i])
        solution1 = min_solution1
        errors1.append(abs(true_energies[i] - umin1))
        solution2 = min_solution2
        errors2.append(abs(true_energies[i] - umin2))
        
        # Translate structure back to origin.
        translate_to_origin(solution1)
        translate_to_origin(solution2)
    
    print(solution1)
    print(solution2)
    print('Energy: %.50f | %.50f' % (umin1, umin2))

    ############################################################################
    # Compare the two structures visually.                                     #
    ############################################################################
    fig = plt.figure()
    
    # First structure
    points = solution1
    x = [x for x in points[::3]]
    y = [y for y in points[1::3]]
    z = [z for z in points[2::3]]

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x, y, z, c='b', s=100)
    ax1.set_title('Tolerance = %.4e' % tol1)
    
    for j in range(1, len(x)):
        for k in range(0, j):
            ax1.plot([x[k], x[j]], [y[k], y[j]], [z[k], z[j]], color='k', alpha=0.2)
   
    ax1.set_xlim3d(-2, 2)
    ax1.set_ylim3d(-2, 2)
    ax1.set_zlim3d(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Second structure
    points = solution2
    x = [x for x in points[::3]]
    y = [y for y in points[1::3]]
    z = [z for z in points[2::3]]

    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(x, y, z, c='b', s=100)
    ax1.set_title('Tolerance = %.4e' % tol2)
    for j in range(1, len(x)):
        for k in range(0, j):
            ax1.plot([x[k], x[j]], [y[k], y[j]], [z[k], z[j]], color='k', alpha=0.2)

    ax1.set_xlim3d(-2, 2)
    ax1.set_ylim3d(-2, 2)
    ax1.set_zlim3d(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Plot both structures
    fig.suptitle('n = %d\n' % (len(points) // 3) +
                 'Comparing errors.')
    plt.show()

    ############################################################################
    # Compare the two structures' errors.                                      #
    ############################################################################
    fig = plt.figure()
    
    # First structure
    errors = errors1
    ax = fig.add_subplot(121)
    ax.plot([i for i in range(3, len(errors) + 3)], errors)
    ax.scatter([i for i in range(3, len(errors) + 3)], errors, c='r')
    ax.set_title('Error for tol = %.4e' % tol1)
     
    # Only want one label for each axis.
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Forward Error')
    ax.set_yscale('log')
    
    # Second structure
    errors = errors2
    ax = fig.add_subplot(122)
    ax.plot([i for i in range(3, len(errors) + 3)], errors)
    ax.scatter([i for i in range(3, len(errors) + 3)], errors, c='r')
    ax.set_title('Error for tol = %.4e' % tol2)
    ax.set_yscale('log')
    
    plt.show()

def plot_n_structures(n, initial_solution, optimizer, u, num_randoms, **kwargs):
    progress = 0
    solution = initial_solution
    min_solution = None

    fig = plt.figure()
    
    for i in range(3, n + 1):
        progress = 0
        a = 0.75
        umin = 0

        # Try num_randoms many guesses.
        for _ in range(num_randoms):
            intermediate = np.append(solution, [uniform(-a, a),
                                                uniform(-a, a),
                                                uniform(-a, a)])

            while u(intermediate) > -1:
                intermediate = np.append(solution, [uniform(-a, a),
                                                    uniform(-a, a),
                                                    uniform(-a, a)])

            # TODO: Find a cleaner way to do this. 
            # 
            # Decide if we are using a jacobian or derivative free method.
            #print('starting optimizer')
            if 'J' in kwargs:
                v = optimizer(F=u, x0=intermediate, **kwargs)
            else:
                v = optimizer(u, intermediate, **kwargs)

            # Update minimum solution if necessary.
            uv = u(v)
            if umin > uv:
                min_solution = v
                umin = uv
                
            # Update progress bar.
            progress += 1
            print('Done: %.2f%%' % ((progress / num_randoms) * 100), end='\r')
            
        print('--------------------------')
        print('Number of atoms: %d\n' % i + 
              'Estimated Minimum Energy : %10f\n' % umin + 
              'Actual Minimum Energy    : %10f\n' % true_energies[i] + 
              'Absolute error           : %10f\n' % abs(true_energies[i] - umin))

        solution = min_solution
        
        # Translate structure back to origin.
        translate_to_origin(solution)
        
        points = solution[:]
        x = [x for x in points[::3]]
        y = [y for y in points[1::3]]
        z = [z for z in points[2::3]]
        fignum = int('23' + str(i - 2))
        ax1 = fig.add_subplot(fignum, projection='3d')
        ax1.scatter(x, y, z, c='b', s=100)

        for j in range(1, len(x)):
            for k in range(0, j):
                ax1.plot([x[k], x[j]], [y[k], y[j]], [z[k], z[j]], color='k', alpha=0.2)
        ax1.set_title('n = %d' % i)
        ax1.set_xlim3d(-1.5, 1.5)
        ax1.set_ylim3d(-1.5, 1.5)
        ax1.set_zlim3d(-1.5, 1.5)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

    input('Ready to display structures?')
    
    fig.suptitle('Structures %d through %d' %(3, i))
    plt.show()


def translate_to_origin(v):
    x = v[0]
    y = v[1]
    z = v[2]
    
    # Move back to origin
    for i in range(0, len(v), 3):
        v[i] -= x
        v[i + 1] -= y
        v[i + 2]-= z

        
def plot_errors(errors):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(3, len(errors) + 3)], errors)
    ax.scatter([i for i in range(3, len(errors) + 3)], errors, c='r')
    ax.set_title('Lennard-Jones Clusters Absolute Error')
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Forward Error')
    ax.set_yscale('log')
    plt.show()
    
    
def main():
    # First two points, which are trivial.
    initial_solution = np.array([0.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0])
    n = 8     # Number of atoms.
    r = 5000  # Number of random guesses. 
    

    ##                                                                        ##
    #  Try Nelder-Mead                                                         #
    ##                                                                        ##
    '''
    errors, = try_optimizer(n, initial_solution, opt.nelder_mead, u, r,
                           rad=1, max_iter=999, xtol=0.5e-6, ftol=0.5e-6)
    plot_errors(errors)
    '''
    
    ##                                                                        ##
    #  Try steepest descent with weakest_line search                           #
    ##                                                                        ##
    '''
    errors = try_optimizer(n, initial_solution, opt.weakest_line,
                           u, r, J=du, tol=0.5e-10)
    plot_errors(errors)
    '''
    
    ##                                                                        ##
    #  Compare two tolerances                                                  # 
    ##                                                                        ##
    #compare_tolerances(n, initial_solution, u, du, r, 0.5e-8, 0.5e-14)
    
    ##                                                                        ##
    #  Plot all n structures                                                   # 
    ##                                                                        ##
    plot_n_structures(n, initial_solution, opt.weakest_line,
                           u, r, J=du, tol=0.5e-10)
    
    ##                                                                        ##
    #  Try steepest descent with weakest_line search                           #
    #  Using completely random guesses.                                        # 
    ##                                                                        ##
    '''
    errors = try_all_random(n, initial_solution, opt.weakest_line,
                           u, r, J=du, tol=0.5e-10)
    plot_errors(errors)
    '''
    

    ##                                                                        ##
    #  Try steepest descent with golden section search                         #
    ##                                                                        ##
    '''
    errors, = try_optimizer(n, initial_solution, opt.steepest_descent_gss,
                           u, r, J=du, tol=0.5e-10)
    plot_errors(errors)
    '''
    
    ##                                                                        ##
    #  Try conjugate gradient search                                           # 
    ##                                                                        ##
    '''
    errors, = try_optimizer(n, initial_solution, opt.conjugate_gradient_search,
                           u, r, J=du, tol=0.5e-10)
    plot_errors(errors)
    '''


if __name__ == '__main__':
    main()
