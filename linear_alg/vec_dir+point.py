import numpy as np
import matplotlib.pyplot as plt

"""
define a line by 1) a direction - e, 2) a point - P

t is scalar values
L = P + et

"""

def test():
    d = np.array([2, 1])
    e = d/np.linalg.norm(d)

    dline = np.vstack([e * scalar for scalar in range(10)])

    p = np.array([1, 5])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x, y = d
    ax.scatter([x], [y], marker="o", s=50, color="red")
    ax.plot(dline.T[0], dline.T[1], color="green")
    ax.scatter([p[0]], [p[1]], marker="o", s=50, color="blue")

    Y = dline + p
    print Y
    ax.scatter(Y.T[0], Y.T[1], color="magenta", marker="o")

    ax.grid("on")

    plt.savefig("test.jpg")

if __name__ == '__main__':
    test()