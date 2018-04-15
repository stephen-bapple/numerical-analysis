import numpy as np

def u(points):
    """
    Potential function.
    """
    potential = 0

    # The potential is the sum of all the forces between each pair of molecules.
    for j in range(3, len(points), 3):
        for i in range(0, j, 3):
            r = ((points[i] - points[j])**2 + (points[i + 1] - points[j + 1])**2
                 + (points[i + 2] - points[j + 2])**2)
            potential += r**-6 - 2*r**-3

    return potential


def du(points):
    """
    Jacobian for the potential function.
    """

    v = [0] * len(points)
    #for k in range(len(points) // 3):
    for k in range(len(points)):
        potential = 0

        for j in range(3, len(points), 3):
            for i in range(0, j, 3):
                # Check if we're taking the derivative relative to this variable.
                if i <= k <= i + 2:
                    if k == i:
                        jk = j
                    elif k == i + 1:
                        jk = j + 1
                    else:  # k == i + 2
                        jk = j + 2

                    r = ((points[i] - points[j])**2
                         + (points[i + 1] - points[j + 1])**2
                         + (points[i + 2] - points[j + 2])**2)

                    potential += 12*(-r**-7 + r**-4)*(points[k] - points[jk])

                else:
                    r = ((points[i] - points[j])**2
                         + (points[i + 1] - points[j + 1])**2
                         + (points[i + 2] - points[j + 2])**2)
                    potential += r ** -6 - 2 * r ** -3

        v[k] = potential

    #print(v)
    return np.array(v)