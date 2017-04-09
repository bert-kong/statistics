import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def cov2cor(A):

    ss00 = A[0, 0]
    ss11 = A[1, 1]
    ss01 = A[0, 1]

    s0 = np.sqrt(ss00)
    s1 = np.sqrt(ss11)
    s01 = ss01/(s0 * s1)

    return np.matrix([[1, s01], [s01, 1]])


def test(degree=69, scale=[3., .3]):
    size = 2000

    rad = np.radians(degree)

    cos = np.cos(rad)
    sin = np.sin(rad)

    S = np.diag(scale)
    R = np.matrix([[cos, -sin],
                   [sin, cos]])

    COV = R * S * R.T

    X = np.random.multivariate_normal(mean=[0, 0],
                                      cov=COV,
                                      size=size)
    COR = np.corrcoef(X.T)

    Y = np.random.multivariate_normal(mean=[0, 0],
                                      cov=COR,
                                      size=500)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    #colors = cm.plasma()
    ax.scatter(X.T[0], X.T[1], marker="o", color="green", s=1)

    ax.scatter(Y.T[0], Y.T[1], marker="o", color="red", s=3)


    # set attributes of coordinates
    b = 9
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_aspect("equal", adjustable="box")



    plt.savefig("test.jpg")

    print COV
    print COR
    print cov2cor(COV)


if __name__ == '__main__':
    test()