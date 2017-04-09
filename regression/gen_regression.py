import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gen(size=1000, sigma=2, B0=2, B1=1):
    """
    simulate regression data

    population real relationship btw Y~X
    Y = B0 + B1X + e
    e ~ N(0, sigma), independent from X

    sigma, variance, is a constant value through X
    2 rows, Y, X
    :return: numpy array (rows, columns)
    """

    #--------------------------
    # generate error N(0, sigma)
    #--------------------------
    mu = 0
    errors = np.random.normal(loc=mu, scale=sigma, size=size)

    #--------------------------
    # generate X values uniform
    #--------------------------
    X = np.random.uniform(low=-3, high=3, size=size)

    #--------------------------
    # compute Y = B0 + B1X + error

    f = lambda X : B0 + (B1 * X)
    Y = f(X) + errors

    return X, Y


def draw(X, Y, B0=2, B1=1):
    # Y = B0 + B1X
    f = lambda X : B0 + (B1 * X)

    fig = plt.figure()

    # plot linear regression
    index = 121
    ax = fig.add_subplot(index)
    ax.scatter(X, Y, color='g', s=4, marker='o')
    ax.plot(X, f(X), color='r', linestyle='-')
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.grid(True)

    # plot logistic regression
    # mapping Y -> logistic(Y)
    logistic=lambda Y : 1/(1 + np.exp(-Y))

    index += 1
    ax2 = fig.add_subplot(index)
    Z = logistic(Y)
    ax2.scatter(Y, Z, color='blue', marker='o', s=4)
    plt.axhline(y=0)
    plt.axhline(y=.5)
    plt.axvline(x=0)

    plt.grid(True)

    #plt.savefig('test.jpg')


def estimate(X, Y):
    """
    estimate B0, B1 based on samples X, Y
    :param Y:
    :param X:
    :return:
    """

    #-----------------------------------
    # assumption: Y = B0 + B1X + error
    # error - N(0, sigma)
    #-----------------------------------
    mux = X.mean()
    muy = Y.mean()

    SSxy = np.sum(((X - mux) * (Y - muy)))
    SSx = np.sum((X - mux) ** 2)
    b1 = SSxy/SSx

    # b0 = mean(Y) - b1 x mean(X)
    b0 = muy - (b1 * mux)
    print "estimated B0, B1 (2, 1) ~ b0, b1 = %0.2f, %0.2f" % (b0, b1)


def least_square_3d(Xconst, Yconst, b0, b1):

    delta = 10
    B0 = np.arange(b0-delta, b0+delta, .1)
    B1 = np.arange(b1-delta, b1+delta, .1)

    #--------------------------------
    #  Y = B0 + B1X + errors
    #  errors = Y - (B0 + B1X)
    #  find B0 and B1 to minisize:
    #  min(sum((Y - (B0 + B1X)) ** 2))
    #--------------------------------

    # X, Y are coeffients values/constant
    # B0, B1 are variables
    # SSe is function of B0, B1

    BB0, BB1 = np.meshgrid(B0, B1)
    SSe = np.zeros(shape=BB0.shape, dtype=BB0.dtype)

    rows, columns = BB0.shape
    for r in xrange(rows):
        for c in xrange(columns):
            Es = Yconst - BB0[r][c] - (BB1[r][c] * Xconst)
            SSe[r][c] = (np.sum(Es ** 2))/10000.0

    fig = plt.figure()
    index = 111
    ax = fig.add_subplot(index, projection='3d')

    plt.grid(True)
    cstride, rstride = 20, 20
    ax.plot_wireframe(BB0, BB1, SSe, cstride=cstride, rstride=cstride, color='green', label="unit-10000")
    print "label ---> ", ax.get_label()
    ax.set_xlabel("B0")
    ax.set_ylabel("B1")
    ax.set_zlabel("SSE")
    ax.set_zticks([5, 15, 25, 35, 45])
    ax.legend()
    print "ticks ---> ", ax.get_xticks()
    print "ticks Z ---> ", ax.get_zticks()

    #ax.view_init(30, 90)

    plt.savefig('test.jpg')





if __name__ == '__main__':
    B0, B1 = 2., 1.0
    X, Y = gen(B0=B0, B1=B1)

    least_square_3d(X, Y, B0, B1)
    # draw(X, Y, B0, B1)
    #estimate(X, Y)