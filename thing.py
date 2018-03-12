'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm

def pitch(s,x,r):
    x = s[0]
    dxdt = r * x + np.power(x,3)- np.power(x,5)
    return [dxdt]

t = np.arange(0,100,2)
s0=[-50]

N = 200 # Number of points along each side of the diagram
diagram = np.zeros((N,N))

rmin,rmax = -10,10
rrange = np.arange(rmin, rmax,(rmax-rmin)/N)   

smin,smax = -5.0,5.0
srange = np.arange(smin,smax,2*(smax-smin)/N)    

for i in rrange:
    for s0 in srange:
        s = odeint(pitch,[s0],t, args=(i,))
        imgx = int((i-rmin)*N/(rmax-rmin))
        imgy = int((s[-1]-smin)/(smax-smin)*N)
        imgx = min(N-1,max(0,imgx)) # make sure we stay 
        imgy = min(N-1,max(0,imgy)) # within the diagram bounds
        diagram[imgy,imgx] = 1

plt.title("Bifurcation diagram")
plt.plot(np.flipud(diagram))
#plt.imshow(np.flipud(diagram),cmap=cm.Greys,
#           extent=[rmin,rmax,smin,smax],aspect=(rmax-rmin)/(smax-smin))
plt.xlabel("r")
plt.ylabel("x")
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np

m=0.7
# Initialize your data containers identically
X = np.linspace(0.7,4,10000)
Y = []
# l is never used, I removed it.
for x in X:
    # Start with a random value of m instead of remaining stuck
    # on a particular branch of the diagram
    m = np.random.random()
    for n in range(1000):
        m=(x*m)*(1-m)
    Y.append(m)
# Remove the line between successive data points, this renders
# the plot illegible. Use a small marker instead.
plt.plot(X, Y, ls='', marker=',')
plt.show()