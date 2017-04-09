import numpy as np
import matplotlib.pyplot as plt


def transformation(XX1, XX2):
    """
    skew symmetric
    :param X1:
    :param X2:
    :return:
    """
    a = np.random.randint(low=1, high=10, size=1)
    A = np.matrix([0, a, -a, 0]).reshape(2, 2)
    print "skewed matrix"
    print A

    UU = np.zeros(shape=(5, 9))
    VV = np.zeros(shape=(5, 9))

    for i in range(5):
        # create one line(x, y) of the mesh
        L = np.matrix([XX1[i],XX2[i]])

        # transformation to new line
        Lt = A * L

        # assign transformed points back to messh
        UU[i] = Lt[0]
        VV[i] = Lt[1]


    return UU, VV

def set_plot_attr(plot):

    plot.grid("on")
    plot.set_aspect("equal", adjustable="box")
    plot.axhline(y=0, color="blue", linewidth=1)
    plot.axvline(x=0, color="blue", linewidth=1)
    b = 10
    plot.set_xlim(-b, b)
    plot.set_ylim(-b, b)


def test():
    X = np.arange(start=-4, stop=5, step=1)
    Y = np.arange(start=-2, stop=3, step=1)
    XX, YY = np.meshgrid(X, Y)
    print "----------> ", XX.shape
    print "----------> ", YY.shape

    fig = plt.figure()
    index = 221
    ax = fig.add_subplot(index)

    marker_sz = 1
    ax.scatter(XX, YY, s=marker_sz, marker='o', color="green")

    UU, VV = transformation(XX, YY)
    ax.scatter(UU, VV, s=marker_sz, marker='o', color="red")

    # set axis attributes
    set_plot_attr(ax)

    plt.savefig("test.jpg")




if __name__ == '__main__':
    test()