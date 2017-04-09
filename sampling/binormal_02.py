
import numpy as np
import matplotlib.pyplot as ply


def std_normal(size=1000):

    mean = [0, 0]
    cov  = np.matrix([[1, 0], [0, 1]])
    XY = np.random.multivariate_normal(mean, cov, size)

    fig = ply.figure()
    ax = fig.add_subplot(211)

    X = XY[:, 0]
    Y = XY[:, 1]

    b = 5
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.grid("on")
    ax.axhline(y = 0, color="blue", linewidth=2)
    ax.axvline(x = 0, color="blue", linewidth=2)
    ax.set_aspect("equal", adjustable="box")

    ax.scatter(X, Y, color="blue", marker="o", s=.5)

    ply.savefig("test.jpg")


def coord_attributes(ax):
    b = 10
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.grid("on")
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(y=0, color="green", linewidth=2)
    ax.axvline(x=0, color="green", linewidth=2)


def get_cov(x, y, degree):
    sin = np.sin(np.radians(degree))
    cos = np.cos(np.radians(degree))

    R = np.matrix([[cos, -sin], [sin, cos]])
    S = np.diag([x, y])
    cov = R * S * R.I

    return cov

def normal(size=1000):

    fig = ply.figure()

    Sx, Sy = 4, 2
    degree = 30
    cov = get_cov(Sx, Sy, degree)

    #------------------------------------
    # generate normal distribution
    #------------------------------------
    mean = [0, 0]
    XY = np.random.multivariate_normal(mean=mean,
                                       cov = cov,
                                       size=size)

    X = XY[:, 0]
    Y = XY[:, 1]

    ax = fig.add_subplot(121)
    coord_attributes(ax)
    ax.scatter(X, Y, color="blue", marker="o", s=.5)


    #------------------------------------
    # generate standard normal
    #------------------------------------
    std = np.matrix([[1, 0], [0, 1]])
    XY = np.random.multivariate_normal(mean=mean,
                                       cov = std,
                                       size=size)

    cov = get_cov(np.sqrt(Sx), np.sqrt(Sy), degree)
    XY2 = cov * XY.T

    ax2 = fig.add_subplot(122)
    coord_attributes(ax2)

    X = np.array(XY2[0])
    Y = np.array(XY2[1])
    ax2.scatter(X, Y,
                color = "blue",
                marker="o",
                s = .5)


    ply.savefig("test.jpg")


if __name__=='__main__':

    normal()


