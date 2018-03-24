import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# This function should be a plane. It should not change across y since it
# purely depends on x.
def F(x, y):
    return 3 * x

n = 1000
xmin = -2
xmax = 2
ymin = -2
ymax = 2
x = np.linspace(xmin, xmax, n + 1)
y = np.linspace(ymin, ymax, n + 1)

# Observe that this graph is not correct.
# The z value of the plane varies with y instead of x.
Z = np.zeros((n + 1, n + 1))
for i in range(n + 1):
    for j in range(n + 1):
        Z[i, j] = F(x[i], y[j])

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

p1 = ax1.plot_surface(X, Y, Z, cmap=cm.terrain, label="Something's wrong...")

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.show()


# Observe that this graph is what you would expect.
# Swapping the indices fixed the problem.
Z = np.zeros((n + 1, n + 1))
for i in range(n + 1):
    for j in range(n + 1):
        # Rows: y values. Columns: x values
        Z[j, i] = F(x[i], y[j])
        
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

p1 = ax1.plot_surface(X, Y, Z, cmap=cm.terrain, label='Correct')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.show()