
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d



def binorm(X, Y, rho):
    # sqrt(1-q**2)
    p=1 /(2 * np.pi * np.sqrt(1-(rho ** 2)))

    fnorm = lambda X, Y: (1/p) * np.exp(-1/(2 * (1 - rho**2)) * ((X**2) + (Y**2) - (2 * X * Y * rho)))

    return fnorm(X, Y)

def test():
    X = np.arange(-3, 3, .01)
    Y = np.arange(-3, 3, .01)

    XX, YY = np.meshgrid(X, Y, sparse=False)
    ZZ = binorm(XX, YY, .9)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(XX, YY, ZZ, color='#00ff00', cstride=30, rstride=30, linewidth=.5)
    ax.view_init(30, 30)

    ax.contour(XX, YY, ZZ, color='r')

    plt.savefig("test.jpg")


def contour_fill():


    # preparing data
    X = np.arange(-3, 3, .01)
    Y = np.arange(-3, 3, .01)
    XX, YY = np.meshgrid(X, Y, sparse=False)

    ZZ = binorm(XX, YY, .9)

    # drawing contour fill

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    levels = [0.0, .2, .5, .9, 1.5, 2.5, 3.5]
    cp = ax.contourf(XX, YY, ZZ, levels)
    plt.colorbar(cp)
    ax.view_init(-90, 0)

    plt.savefig("test.jpg")




if __name__=='__main__':
    #test()
    contour_fill()
