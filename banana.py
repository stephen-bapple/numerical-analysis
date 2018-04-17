import optimizers as opt

def main():
    '''
    x, y = opt.multivariate_newtons(J, H, [-1.9, 2], 0.5e-15)
    print('(%.20f, %.20f)' % (x, y))
    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])

    x, y = opt.weakest_line(F, J, [-1.9, 2], tolerance=0.5e-15)
    print('(%.20f, %.20f)' % (x, y))
    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])

    x, y = opt.steepest_descent_gss(F, J, [-1.9, 2], tolerance=0.5e-15)
    print('(%.20f, %.20f)' % (x, y))
    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])
    '''
#    v = opt.nelder_mead(F, [-1.9, 2], 3, xtol=0.5e-10, ftol=0.5e-10)
#    print(v)
#    x = v[0]
#    y = v[1]
#    print('(%.20f, %.20f)' % (x, y))
#    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])

    v = opt.conjugate_gradient_search(F, J, [-1.9, 2])    
    x = v[0]
    y = v[1]
    print('(%.20f, %.20f)' % (x, y))
    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])
#    
#    v = opt.weakest_line(F, J, [-1.9, 2])
#    x = v[0]
#    y = v[1]
#    print('(%.20f, %.20f)' % (x, y))
#    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])

def F(v):
    """
    Rosenbrock's "banana function"
    """
    x, y = v
    return (1 - x)**2 + 100 * (y - x**2)**2


def J(v):
    x, y = v
    return [2*x - 2 - 400*x*(y - x**2), 200*(y - x**2)]


def H(v):
    x, y = v
    return [[2 - 400*y + 1200*x**2, -400*x],
            [-200, 200]]


if __name__ == '__main__':
    main()
