import numpy as np
import matplotlib.pyplot as plt


def generate_data(size):

    X1 = np.linspace(-1, 1, size, endpoint=True)
    X2 = np.sqrt(1 - (X1**2))

    X = np.concatenate([X1, X1])
    Y = np.concatenate([X2, -X2])

    #return np.array([X1, X2])
    return np.array([X, Y])


def create_matrix(degree=74, scale=(2, .4)):

    #----------------------------------
    #  create matrix
    #----------------------------------
    radian = np.deg2rad(degree)

    sin = np.sin(radian)
    cos = np.cos(radian)
    R = np.matrix([[cos, -sin], [sin, cos]])
    S = np.diag(scale)

    return R, S


def transformation(X, COV):
    values, vectors = np.linalg.eig(COV)

    return COV * X


def config_plot_attr(ax):
    ax.grid("on")
    ax.axhline(y=0, color="blue", linewidth=2)
    ax.axvline(x=0, color="blue", linewidth=2)

    b = 5
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)

    ax.set_aspect("equal", adjustable="box")


def draw_01():
    fig = plt.figure()

    index = 221
    ax = fig.add_subplot(index)

    X = generate_data(10)

    R, S = create_matrix(degree=67, scale=(2, .3) )
    COV = R * S * R.transpose()
    Y = transformation(X, COV)

    #ax.scatter(X[0, :], X[1, :], color="green", marker="o", s = 5)
    ax.scatter(Y[0, :], Y[1, :], color="green", marker="o", s = 5)

    p1 = np.array([1, 2.5])
    p2 = np.array([2, 1])

    mu1 = p1.reshape(2, 1)
    mu2 = p2.reshape(2, 1)
    mu = np.concatenate([mu1.transpose(), mu2.transpose()]).transpose()
    midpoint = mu.sum(axis=1)/2

    p = p2 - p1
    a = COV.I * p.reshape(2,1)


    U = Y + mu1
    V = Y + mu2
    #ax.scatter(U[0, :], U[1, :], color="red", marker="o", s = 5)
    #ax.scatter(V[0, :], V[1, :], color="magenta", marker="o", s = 5)

    #ax.scatter(mu[0, :], mu[1, :], color="black", marker="o", s = 15)

    config_plot_attr(ax)

    #------------------------------------------------
    #   2nd plot
    #------------------------------------------------
    index += 1
    ax2 = fig.add_subplot(index)
    ax2.scatter(U[0, :], U[1, :], color="red", marker="o", s = 5)
    ax2.scatter(V[0, :], V[1, :], color="magenta", marker="o", s = 5)

    ax2.scatter(mu[0, :], mu[1, :], color="black", marker="o", s = 15)
    ax2.scatter(midpoint[0], midpoint[1], color="black", marker="o", s = 15)

    # the optimum direction: (mean1-mean2)/sigma = max
    ax2.scatter(a[0], a[1], color="red", marker="o", s = 5)

    config_plot_attr(ax2)

    #------------------------------------------------
    #   3nd plot
    #------------------------------------------------
    index += 1
    ax3 = fig.add_subplot(index)

    W = U - mu1
    Z = V - mu2

    ax3.scatter(W[0, :], W[1, :], color="red", marker="o", s = 5)
    ax3.scatter(Z[0, :], Z[1, :], color="magenta", marker="o", s = 5)


    #ax3.scatter(mu[0, :], mu[1, :], color="black", marker="o", s = 15)

    config_plot_attr(ax3)

    #------------------------------------------------
    #   4th
    #------------------------------------------------
    index += 1
    ax4 = fig.add_subplot(index)
    Q = R.transpose() * W + mu1
    P = R.transpose() * W + mu2
    print "--------- Vectors -----------"
    print R.transpose()

    ax4.scatter(Q[0, :], Q[1, :], color="red", marker="o", s = 5)
    ax4.scatter(P[0, :], P[1, :], color="magenta", marker="o", s = 5)
    ax4.scatter(mu[0, :], mu[1, :], color="black", marker="o", s = 15)


    config_plot_attr(ax4)




    plt.savefig("test.jpg")


def draw_02():
    pass

if __name__ == '__main__':
    draw_01()

