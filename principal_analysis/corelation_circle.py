import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def test(degree=69, scale=[3., .3]):
    rad = np.radians(degree)

    cos = np.cos(rad)
    sin = np.sin(rad)

    S = np.diag(scale)
    R = np.matrix([[cos, -sin],
                   [sin, cos]])

    COV = R * S * R.T

    ss01 = COV[0, 1]
    ss00 = COV[0, 0]
    ss11 = COV[1, 1]

    ss = ss01/(np.sqrt(ss00) * np.sqrt(ss11))
    COR = np.matrix([[1, ss],
                     [ss, 1]])

    size = 20
    X1 = np.linspace(-1, 1, size, endpoint=True)
    X2 = np.sqrt(1 - X1 * X1)
    X1 = np.hstack((X1, X1))
    X2 = np.hstack((X2, -X2))
    X = np.vstack((X1, X2))


    fig = plt.figure()
    ax = fig.add_subplot(111)

    #colors = cm.plasma()
    ax.scatter(X1, X2, marker="o", color="green", s=3)

    Y = COV * X
    ax.scatter(Y[0], Y[1], marker="o", color="red", s=3)

    Z = COR * X
    ax.scatter(Z[0], Z[1], marker="o", color="black", s=3)


    # set attributes of coordinates
    b = 9
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_aspect("equal", adjustable="box")



    plt.savefig("test.jpg")

    print COV
    print COR


if __name__ == '__main__':
    test()