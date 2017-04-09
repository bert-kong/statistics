import numpy as np
import matplotlib.pyplot as plt


"""
  draw a line which is perpendicular to
  a vector w = (4, 1)
  a point p = (1, 2)
"""


def test():

    w = np.array([4, 1])
    x, y = w
    d = np.array([y, -x])
    e = d/np.linalg.norm(d)

    # dot(w, dline) = 0
    dline = np.vstack([e * scalar for scalar in range(10)])

    # vector w
    w = w/np.linalg.norm(w)
    wline = np.vstack([w * scalar for scalar in range(10)])

    # Point
    p = np.array([1, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #x, y = d
    #ax.scatter([x], [y], marker="o", s=50, color="red")
    #ax.plot(dline.T[0], dline.T[1], color="green")
    ax.plot(wline.T[0], wline.T[1], color="green")
    x, y = 4, 1
    ax.annotate("w vector",
                xy=(x, y),
                xytext=(x+1, y+1),
                arrowprops=dict(facecolor="black", shrink=.01),)

    # Point (x, y)
    ax.scatter([p[0]], [p[1]], marker="o", s=70, color="blue")

    #-------------------------------------
    # draw line
    #-------------------------------------
    Y = dline + p

    x, y = Y.T[:, -1]
    ax.scatter(Y.T[0], Y.T[1], color="magenta", marker="o")

    x, y = p
    ax.annotate("line passes through (%d, %d)" % (1, 2),
                xy = (x, y),
                xytext=(x+1, y+1),
                arrowprops=dict(facecolor="black", shrink=.1))

    #-------------------------------------
    ax.grid("on")
    ax.axhline(y=0, color="blue")
    ax.axvline(x=0, color="blue")
    ax.set_xlim(-2, 5)
    ax.set_ylim(-10, 5)
    ax.set_aspect("equal", adjustable="box")

    plt.savefig("test.jpg")

if __name__ == '__main__':
    test()