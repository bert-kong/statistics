import numpy as np
import matplotlib.pyplot as plt


def test(degree=65):

    #------------------------------
    #  Rotation vectors
    #------------------------------
    rad = np.radians(degree)
    sin = np.sin(rad)
    cos = np.cos(rad)

    #-----------------------------------
    #  Vectors
    #-----------------------------------
    V1 = np.array([cos, -sin])
    V2 = np.array([sin, cos])

    # vector lines
    v1 = np.vstack((-V1, V1))
    v2 = np.vstack((-V2, V2))


    #-----------------------------------
    # a horizontal line
    #-----------------------------------
    nr_points = 5
    X1 = np.linspace(0, 2, nr_points, endpoint=True)
    X2 = np.zeros(shape=(5, ), dtype=np.float32)
    X = np.array([X1, X2])

    #-----------------------------------
    #  X projection on V1 and V2
    #-----------------------------------
    Y1 = np.dot(V1, X)
    Y2 = np.dot(V2, X)
    Y = np.array([Y1, Y2])


    fig = plt.figure()
    ax = fig.add_subplot(111)

    #---------------------------------------------------
    #  Draw a new coordinates (rotated)
    #---------------------------------------------------
    Dx, Dy = v1.T[0], v1.T[1]
    ax.plot(Dx, Dy, linestyle='-', color='green')

    ax.annotate("Y1", xy = (cos, -sin), xycoords="data")

    Dx, Dy = v2.T[0], v2.T[1]
    ax.plot(Dx, Dy, linestyle='-', color='red')
    ax.annotate("Y2", xy=(sin, cos), xycoords="data")

    # rotation for X-axis
    ax.scatter(Y1, Y2, marker=".", color="black")

    #---------------------------------------------------
    # Y1 and Y2 are projections(scalar) on V1 and V2 respectively
    #---------------------------------------------------

    Z1 = np.vstack([V1 * x for x in Y1])
    Z2 = np.vstack([V2 * x for x in Y2])

    ax.scatter(Z1.T[0], Z1.T[1], marker='o', color="red")
    ax.scatter(Z2.T[0], Z2.T[1], marker='o', color="blue")

    # points on Axis - X
    ax.scatter(X1, X2, marker='o', color="magenta")

    #---------------------------------------------------
    # plot attributes
    #---------------------------------------------------
    ax.grid("on")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(y=0, color="blue", linewidth=1)
    ax.axvline(x=0, color="blue", linewidth=1)

    plt.savefig("test.jpg")

if __name__ == '__main__':
    test(degree=40)