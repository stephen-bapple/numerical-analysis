# Program 13.3 Nelder-Mead Search
#
# Program copied from Numerical Analysis 2nd Edition by 
# Timothy Sauer.
# And translated from Matlab to Python by Stephen Bapple
#
# Input: function f, best guess xbar (column vector),
# initial search radius rad and number of steps k
# Output: matrix x whose columns are vertices of simplex,
# function values y of those vertices


import numpy as np


def nelder_mead(f,xbar,rad,k):
    xbar = np.array(xbar, dtype=float)
    n = xbar.shape[0]
    print(n)
    x = np.array((n, n + 1))
    y = np.array((1, n + 1))

    # Done?
    # maybe: 
    # x = np.array((n, n + 1))
    # y 
    x[:,0] = xbar; # each column of x is a simplex vertex
    x[:, 1:n + 1] = xbar * np.ones((1,n)) + rad * np.identity(n)

    for j in range(0, n + 1):
        y[j]=f(x[:,j])  # evaluate obj function f at each vertex

    #[y,r]=sort(y); # sort the function values in ascending order
    r = y.argsort(axis=0)
    y = y[r]

    x = x[:, r] # and rank the vertices the same way

    for i in range(0, k + 1):
        xbar = np.mean[x[:, 0:n].T].T # xbar is the centroid of the face
        xh = x[:, n]          # omitting the worst vertex xh
        xr = 2*xbar - xh 
        yr = f(xr)
        
        if yr < y[n - 1]
            if yr < y[0] # try expansion xe
                xe = 3*xbar - 2*xh 
                ye = f(xe)
                
                if ye < yr # accept expansion
                    x[:, n] = xe
                    y[n] = f(xe)
                else #% accept reflection
                    x[:,n] = xr 
                    y[n] = f(xr)

            else # xr is middle of pack, accept reflection
                x[:,n] = xr
                y[n] = f(xr)
        else # xr is still the worst vertex, contract
            if yr < y[n] # try outside contraction xoc
                xoc = 1.5*xbar - 0.5*xh
                yoc = f(xoc)
                if yoc < yr # accept outside contraction
                    x[:,n] = xoc
                    y[n] = f(xoc)
                else # shrink simplex toward best point
                    for j in range(1, n + 1)
                        x[:,j] = 0.5*x[:,0]+0.5*x[:,j] 
                        y[j] = f(x[:,j])
            
            else # xr is even worse than the previous worst
                xic = 0.5*xbar+0.5*xh
                yic = f(xic)
                
                if yic < y[n] # accept inside contraction
                    x[:,n] = xic 
                    y[n] = f(xic)
                else # shrink simplex toward best point
                    for j in range(1, n + 1):
                        x[:,j] = 0.5*x[:,1]+0.5*x[:,j]
                        y[j] = f(x[:,j])
        
        # Resort the obj function values.
        r = y.argsort(axis=0)
        y = y[r]
        x = x[:, r] # and rank the vertices the same way
    return x