import optimizers as opt


def F(x, y):
    """
    Rosenbrock's "banana function"
    """
    return (1 - x)**2 + 100 * (y - x**2)**2


def J(x, y):
    return [2*x - 2 - 400*x*(y - x**2), 200*(y - x**2)]


def H(x, y):
    return [[2 - 400*y + 1200*x**2, -400*x],
            [-200, 200]]


if __name__ == '__main__':
    x, y = opt.multivariate_newtons(J, H, [-1.9, 2], 0.5e-15)
    print('(%.20f, %.20f)' % (x, y))
    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2],[x, y]])

    x, y = opt.weakest_line(F, J, [-1.9, 2], tolerance=0.5e-15)
    print('(%.20f, %.20f)' % (x, y))
    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])

    x, y = opt.steepest_descent_gss(F, J, [-1.9, 2], tolerance=0.5e-15)
    print('(%.20f, %.20f)' % (x, y))
    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])

    x, y = opt.weakest_line(F, J, [-1.9, 2], tolerance=0.5e-15)
    print('(%.20f, %.20f)' % (x, y))
    opt.plot3d_with_mins(F, [-2, 2], [-1, 3], [[-1.9, 2], [x, y]])
