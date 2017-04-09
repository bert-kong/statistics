import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from mpl_toolkits.mplot3d import Axes3D



def linestyle():
    X=np.linspace(-3, 3, 3)
    Y=np.linspace(-3, 3, 4)


    XX, YY = np.meshgrid(X, Y, sparse=False)
    print XX.shape, YY.shape

    ZZ = np.sqrt(XX**2 + YY**2)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #cset = ax.contour(XX, YY, ZZ, color='green')
    cset = ax.contour(XX, YY, ZZ, colors='green', linestyles='dashed')
    ax.clabel(cset, inline=True, fontsize=10)

    ax.view_init(90, 90)

    ax.set_title("Contour Plot")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    plt.savefig("test.jpg")


def filled_contours():
    X = np.linspace(-3, 3, 100)
    Y = np.linspace(-3, 3, 100)

    XX, YY = np.meshgrid(X, Y, sparse=False)
    ZZ = np.sqrt(XX**2 + YY**2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # contour fill
    cp=ax.contourf(XX, YY, ZZ)
    plt.colorbar(cp)

    ax.view_init(90, 0)
    plt.savefig("test.jpg")



if __name__ == '__main__':
    filled_contours()