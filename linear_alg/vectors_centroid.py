import numpy as np
import matplotlib.pyplot as plt


def set_attr(ax):

    ax.grid("on")
    ax.axhline(y=0, color="blue")
    ax.axvline(x=0, color="blue")

    b = 10
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_aspect("equal", adjustable="box")


def test():

    A = np.array([1, 4])
    B = np.array([2, 2])

    C = A + 3*(B-A)

    points = np.vstack((A, B, C))

    fig = plt.figure()
    ax = fig.add_subplot(111)


    ax.scatter(points.T[0],
               points.T[1],
               color="red",
               marker="o")

    x, y = A
    ax.annotate("A", xy=(x+1, y+1), xycoords="data")

    x, y = B
    ax.annotate("B", xy=(x+1, y+1), xycoords="data")

    x, y = C
    ax.annotate("C", xy=(x+1, y+1), xycoords="data")

    set_attr(ax)

    plt.savefig("test.jpg")



if __name__ == '__main__':
    test()