'''
Project #2 for Numerical Analysis II

This project revolves around applying various optimization techniques to the
following system: TBD

Optimization techniques used are found in optimizers.py

Author: Stephen Bapple

'''

import numpy as np
import optimizers as opt


def F(x, y):
    '''
    Rosenbrock's "banana function"
    '''
    return (1 - x)**2 + 100 * (y - x**2)**2
    
def J(x, y):
    return -1

def H(x, y):
    return -1

if __name__ == '__main__':
    print(opt.multivariate_newtons())