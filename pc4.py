import numpy as np
from matplotlib import pyplot

def am3_step(f,t,w,h):
    return w[2]+(h/24)*(9*f(t[3],w[3])+19*f(t[2],w[2])-5*f(t[1],w[1])+f(t[0],w[0]))

def ab4_step(f,t,w,h):
    return w[3]+(h/24)*(55*f(t[3],w[3])-59*f(t[2],w[2])+37*f(t[1],w[1])-9*f(t[0],w[0]))

def rk4(f,t,w,h):
    s1 = f(t,w)
    s2 = f(t+h/2,w+(h/2)*s1)
    s3 = f(t+h/2,w+(h/2)*s2)
    s4 = f(t+h,w+h*s3)
    return w+(h/6)*(s1+2*s2+2*s3+s4)

def ytrue(t):
    return 3*np.exp(0.5*t**2)-t**2-2

def f(t,w):
    return t*w+t**3

I = [0,2]
y0 = 1
n = 100

h = (I[1]-I[0])/n

t = [I[0]+i*h for i in range(n+1)]

# setup initial conditions
w = np.zeros((n+1))
w[0] = y0

for i in range(0,3):
    w[i+1] = rk4(f,t[i],w[i],h)

print('size t=%d' % len(t))
print('n=%d' % n)
for i in range(3,n):
    w[i+1] = ab4_step(f,t[i-3:i+1],w[i-3:i+1],h) # predictor
    w[i+1] = am3_step(f,t[i-2:i+2],w[i-2:i+2],h) # corrector

y = [y0]
err = [abs(y[0]-w[0])]
for i in range(1,n+1):
    y.append(ytrue(t[i]))
    err.append(abs(y[i]-w[i]))

print(h)
print(h**4)

fig,ax = pyplot.subplots(2, sharex=True)
ax[0].plot(t,w,label='Approximation')
ax[0].plot(t,y,label='Solution')
ax[0].legend()
ax[0].set_ylim([-1,20])
ax[0].set_xlim(I)
ax[1].plot(t,err)
ax[0].set_ylabel('y')
ax[1].set_ylabel('error')
ax[1].set_xlabel('t')
ax[1].set_yscale('log')
pyplot.show()


