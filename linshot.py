import numpy as np
from matplotlib import pyplot

def rk4_step(f,t,w,h):
    s1 = f(t,w)
    s2 = f(t+h/2,w+(h/2)*s1)
    s3 = f(t+h/2,w+(h/2)*s2)
    s4 = f(t+h,w+h*s3)
    return w + (h/6)*(s1+2*s2+2*s3+s4)

def uf(t,u):
    return np.array([u[1],(2/t**2)*u[0]-(2/t)*u[1]+np.sin(np.log(t))/t**2])

def vf(t,v):
    return np.array([v[1],(2/t**2)*v[0]-(2/t)*v[1]])

def ytrue(t):
    c2 = (8-12*np.sin(np.log(2))-4*np.cos(np.log(2)))/70
    c1 = 1.1 - c2
    return c1*t+c2/(t**2)-.3*np.sin(np.log(t))-.1*np.cos(np.log(t))

I = [1,2]
n = 10
ya = 1
yb = 2

u = np.zeros((n+1,2))
v = np.zeros((n+1,2))
w = np.zeros((n+1))
y = np.zeros((n+1))
err = np.zeros((n+1))

h = (I[1]-I[0])/n

t = [I[0]+i*h for i in range(n+1)]

u[0,:] = [ya,0]
v[0,:] = [0,1]

for i in range(n):
    u[i+1,:] = rk4_step(uf,t[i],u[i,:],h)
    v[i+1,:] = rk4_step(vf,t[i],v[i,:],h)

for i in range(n+1):
    w[i] = u[i,0]+(yb-u[n,0])*v[i,0]/v[n,0]
    y[i] = ytrue(t[i])
    err[i] = abs(y[i]-w[i])

print('\n')
print(' t  |      u     |      v     |      w     |      y     |    err')
print('----+------------+------------+------------+------------+-----------')
for i in range(n+1):
    print('%1.1f | %1.8f | %1.8f | %1.8f | %1.8f | %1.4e' %
            (t[i],u[i,0],v[i,0],w[i],ytrue(t[i]),abs(ytrue(t[i])-w[i])))
print('\n')

fig,ax = pyplot.subplots(2, sharex=True)
ax[0].plot(t,u[:,0],label='IVP_1')
ax[0].plot(t,v[:,0],label='IVP_2')
ax[0].plot(t,w,label='Approximation')
#ax[0].plot(t,y,label='Solution')
ax[0].legend()
ax[0].set_ylim([-0.1,2.1])
ax[0].set_xlim(I)
ax[1].plot(t,err)
ax[0].set_ylabel('y')
ax[1].set_ylabel('error')
ax[1].set_xlabel('t')
ax[1].set_yscale('log')
pyplot.show()
