import numpy as np
import matplotlib.pyplot as plt



def circle():
    """
    generate standard circle points
    :return:
    """

    size = 20
    X1 = np.linspace(-1, 1, size)
    X2 = np.sqrt(1-(X1**2))

    return X1, X2


def draw_standard_circle(ax, X1, X2):
    ax.plot(X1, X2, color='green', linewidth=1, linestyle="-")
    ax.plot(X1, -X2, color='green', linewidth=1, linestyle="-")


def transformation(X1, X2):
    # singlar transformation
    x, y = 2.3, 1.5
    f = 1.2
    A = np.matrix([x, y, f*x, f*y]).reshape(2, 2)

    X = np.matrix([X1, X2])

    Y = A * X
    Y1 = np.array(Y[0])[0]
    Y2 = np.array(Y[1])[0]

    return Y1, Y2


def set_axies_attr(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.grid("on")
    b = 4
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)

    ax.axvline(x=0, color="blue", linewidth=1)
    ax.axhline(y=0, color="blue", linewidth=1)


def test():

    fig = plt.figure()
    index = 221
    ax = fig.add_subplot(index)

    # stantard circle
    X1, X2 = circle()
    draw_standard_circle(ax, X1, X2)


    #---------------------------
    # upper half circle
    #---------------------------
    Y1, Y2 = transformation(X1, X2)
    ax.plot(Y1, Y2, color='red', linewidth=1, linestyle="-")

    #---------------------------
    # bottom half circle
    #---------------------------
    Y1, Y2 = transformation(X1, -X2)
    ax.plot(Y1, Y2, color='black', linewidth=1, linestyle="-")

    #-------------------------------------
    # set coordinates attributes
    #-------------------------------------
    set_axies_attr(ax)


    #-------------------------------------
    # 2nd plot
    #-------------------------------------
    index += 1
    ax2 = fig.add_subplot(index)
    draw_standard_circle(ax2, X1, X2)

    Y1, Y2 = transformation(X1, X2)

    s = [1]
    ax2.scatter(Y1, Y2, s=s, color='red')
    set_axies_attr(ax2)

    #-------------------------------------
    # 2nd plot
    #-------------------------------------
    index += 1
    ax3 = fig.add_subplot(index)
    draw_standard_circle(ax3, X1, X2)
    Y1, Y2 = transformation(X1, -1 * X2)
    #ax3.plot(Y1, Y2, "bp")
    s = [1.]
    ax3.scatter(Y1, Y2, s=s, color="black")
    set_axies_attr(ax3)


    plt.savefig("test.jpg")


if __name__ == '__main__':
    test()