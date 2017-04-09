import numpy as np
import matplotlib.pyplot as plt


def set_axis_attr(plot):

    plot.grid("on")
    b = 10
    plot.set_xlim(-b, b)
    plot.set_ylim(-b, b)

    plot.axhline(y=0, color="blue")
    plot.axvline(x=0, color="blue")


def transformation(X1, X2, degree, symmetric=False):
    radian = np.deg2rad(degree)
    sin = np.sin(radian)
    cos = np.cos(radian)

    R = np.matrix([cos, -sin, sin, cos]).reshape(2, 2)
    print "---- Rotation ----"
    print R

    S = np.diag([5, 9])
    print "---- Scalar ----"
    print S

    X = np.array([X1, X2])

    if symmetric:
        A = R * S * R.transpose()
    else:
        A = R * S


    print "---- Transformation -----"
    print A
    print "---- Eigen Vectors ------"
    print np.linalg.eig(A)

    Y = A * X

    Y1 = np.array(Y[0])[0]
    Y2 = np.array(Y[1])[0]

    return Y1, Y2

def test(degree=45):
    radian = np.deg2rad(degree)
    sin = np.sin(radian)
    cos = np.cos(radian)
    r = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #-------------------------------
    # angle line
    #-------------------------------
    X1 = np.linspace(0, 1, 50)
    X2 = (sin/cos) * X1
    ax.plot(X1, X2, color="green", linestyle="-")

    #-------------------------------
    # horizontal line
    #-------------------------------
    Y1 = X1
    Y2 = np.zeros(shape=(50,))

    ax.plot(Y1, Y2, color="red", linestyle="-", linewidth=2)

    #-------------------------------
    #  transformation R * S
    #-------------------------------
    # transformed angle line
    U1, U2 = transformation(X1, X2, degree)
    ax.plot(U1, U2, color="green", linestyle="-.")

    V1, V2 = transformation(Y1, Y2, degree)
    ax.plot(V1, V2, color="red", linestyle="-.")

    set_axis_attr(ax)
    plt.savefig("test.jpg")



if __name__ == '__main__':
    test(50)