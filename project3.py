'''
Project #3 for Numerical Analysis II

Comparison of different singular value decomposition methods.

Stephen Bapple

'''
import scipy.linalg as la
import numpy as np
from shifted_qr import shifted_qr
from matplotlib import pyplot as plt

ɛ = 0.5e-15

def naive_svd(A, tol=0.5e-15):
    # Ensure A is a numpy matrix.
    A = np.matrix(A)
    
    C = A.T * A
    u = np.matrix(np.zeros((A.shape[0], A.shape[0])))
    sigma = np.matrix(np.zeros(A.shape))
    
    # Get the eigenvalues of A.T * A
    eig = np.linalg.eig(C)    
    
    # Get the singular values
    vals = [s**0.5 for s in eig[0] if abs(s) > tol]
    
    # Insert the singular values into the sigma matrix
    sigma[:len(vals),:len(vals)] = np.diagflat(vals)
    
    # Get the eigenvectors
    v = eig[1]
    
    # Get the columns of u
    for i, val in enumerate(vals):
        u[:,i] = (1 / val) * A * v[:,i] 

    # TODO: 
    # Make sure u is orthonormal?
    
    return u, sigma, v.T

    
def approx_inverse(A, tol=0.5e-15):
    # Ensure A is a numpy matrix.
    A = np.matrix(A)
    
    C = A.T * A
    u = np.matrix(np.zeros((A.shape[0], A.shape[0])))
    sigma = np.matrix(np.zeros(A.shape))
    
    # Get the eigenvalues of A.T * A
    eig = np.linalg.eig(C)    
    
    # Get the singular values
    vals = [s**0.5 for s in eig[0] if abs(s) > tol]
    inv_vals = [1 / s for s in vals if abs(s) > tol]
    
    # Insert the singular values into the sigma matrix
    sigma[:len(vals),:len(vals)] = np.diagflat(inv_vals)
    
    # Get the eigenvectors
    v = eig[1]
    
    # Get the columns of u
    for i, val in enumerate(vals):
        u[:,i] = (1 / val) * A * v[:,i] 

    # TODO: 
    # Make sure u is orthonormal?
    
    return v * sigma * u.T

    
# Return a list of matrices, all the same size, in order of increasing
# condition numbers.
#
# c_lim specifies the lower bound for the condition
# numbers, to ensure that the matrices are 'bad.'
#
# inc is intended to ensure that the condition numbers are increasing,
# but does not guarantee there is any specified gap between them
# due to the random generation of matrices.
def get_bad_matrices_same_size(size, number, c_lim=200, inc=100):
    bad_matrices = []
    condition_limit = c_lim
    
    for _ in range(0, number):

        A = np.matrix(np.random.rand(size, size))

        while np.linalg.cond(A) < condition_limit or\
              np.linalg.cond(A) > condition_limit + inc:
            A = np.matrix(np.random.rand(size, size))

        bad_matrices.append((np.linalg.cond(A), A))
        condition_limit += inc
    
    bad_matrices.sort(key=lambda x: x[0])
    
    #for (a, b) in bad_matrices:
    #    print(a, b.shape)
        
    return bad_matrices


# Return a list of matrices, increasing in size,
# with approximately the same condition number.
def get_bad_matrices_increasing_size(start, stop, c_lim=200, ctol=10):
    bad_matrices = []
    
    for i in range(start, stop + 1):
        
        a = np.matrix(np.random.rand(i, i))

        while np.linalg.cond(a) < c_lim or np.linalg.cond(a) > c_lim + ctol:
            a = np.matrix(np.random.rand(i, i))
        bad_matrices.append((np.linalg.cond(a), a))
    
    #for (a, b) in bad_matrices:
    #    print(a, b.shape)

    return [bad for (_, bad) in bad_matrices]


def f_error(A, inv_A):
    return np.linalg.norm(A * inv_A - np.eye(A.shape[0], A.shape[1]),2)


def compare_sizes(min_size, max_size, lim=200, tol=10):
    matrices = get_bad_matrices_increasing_size(min_size, max_size, 
                                                c_lim=lim, ctol=tol)
    x = [m.shape[0] for m in matrices]
    y = [f_error(m, approx_inverse(m)) for m in matrices]
    y2 = [f_error(m, np.linalg.pinv(m)) for m in matrices]
    fig = plt.figure()
    
    # Plot size vs error
    ax = fig.add_subplot(111)
    ax.plot(x, y, ls='-', label='Custom SVD')
    ax.plot(x, y2,ls=':', label='NumPy pseudo inverse')
    ax.set_xlabel('Size of square matrix')
    ax.set_ylabel('Forward Error: ||inv(A)*A-I||2')
    ax.set_yscale('log')
    ax.set_title('Matrix Size vs. Forward Error for Pseudo inverse\n\
              Square Sizes in [%d, %d]\n\
              Allowed Condition Number Variance: %d' %(min_size, max_size, tol))
    ax.legend()
    plt.show()


def compare_condition(size, number):
    matrices = get_bad_matrices_same_size(size, number)
    x = [m[0] for m in matrices]
    y = [f_error(m[1], approx_inverse(m[1])) for m in matrices]
    y2 = [f_error(m[1], np.linalg.pinv(m[1])) for m in matrices]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(x, y, label='Custom SVD')
    ax.plot(x, y2, label='NumPy pseudo inverse')
    
    ax.set_xlabel('Condition Number')
    ax.set_ylabel('Forward Error: ||inv(A)*A-I||2')
    ax.set_yscale('log')
    ax.set_title('Condition Number vs. Forward Error for Pseudo Inverse\n\
           Matrix size: %d \n\
           # Matrices: %d' % (size, number))
    plt.show()


def main():
    A = np.matrix([[665,666],[666,667]])
    inv_A = approx_inverse(A)
    print('-' * 80)
    print('\n~A^-1:\n', inv_A)
    print('-' * 80)
    print('\nA * ~A^-1\n', A * inv_A)
    print('-' * 80)
    print('Condition(A) =', np.linalg.cond(A))
    print('-' * 80)
    print('Forwards Error: %.10f' % f_error(A, inv_A))
    print('Forwards Error: %.10f' % f_error(A, np.linalg.pinv(A)))
    '''
    print('-' * 80)
    print('Comparing matrices with different sizes.')
    compare_sizes(2, 50)
    
    print('-' * 80)
    print('Comparing matrices with different condition numbers.')
    compare_condition(25, 48)

    print('-' * 80)
    print('Comparing larger matrices with different condition numbers.')
    compare_condition(50, 48)
    
    print('-' * 80)
    print('Comparing matrices with differing sizes and closer condition numbers.')
    compare_sizes(2, 50, tol=5)
    
    print('-' * 80)
    print('Comparing matrices with differing sizes and even closer condition numbers.')
    compare_sizes(2, 50, tol=2)
    '''
    
    print('-' * 80)
    print('Comparing matrices with a huge difference in sizes.')
    compare_sizes(2, 100)

if __name__ == '__main__':
    main()