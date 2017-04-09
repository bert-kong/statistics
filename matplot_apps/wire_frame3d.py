
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def test():

    # create a figure to hold plots
    fig = plt.figure()

    # get a subplot object, in this ex only one plot/graph
    ax = fig.add_subplot(111, projection='3d')

    # create X, Y and Z
    X=np.arange(-1, 1, .01)
    Y=np.arange(-1, 1, .01)

    # create mesh grid
    XX, YY = np.meshgrid(X, Y, sparse=True)

    # standard bivariate normal
    bynorm=lambda X, Y: (1/(2*np.pi)) * np.exp(-.5 * (np.power(X, 2) + np.power(Y, 2)))

    ZZ=bynorm(XX, YY)

    plt.grid(True)
    ax.plot_wireframe(XX, YY, ZZ, rstride=10, cstride=10)

    plt.savefig("test.jpg")


def transform(rho=[0, .3, .6, .9]):

    def binorm(X, Y, rho):
        # sqrt(1-q**2)
        p=1 /(2 * np.pi * np.sqrt(1-(rho ** 2)))

        fnorm = lambda X, Y: (1/p) * np.exp(-1/(2 * (1 - rho**2)) * ((X**2) + (Y**2) - (2 * X * Y * rho)))

        return fnorm(X, Y)


    X=np.arange(-3, 3, .01)
    Y=np.arange(-3, 3, .01)

    XX, YY = np.meshgrid(X, Y, sparse=False)

    fig = plt.figure()

    rstride, cstride = 25, 25

    #--------------------------
    # first plot
    #--------------------------
    index=221
    ax = fig.add_subplot(index, projection='3d')
    plt.grid(True)

    ZZ=binorm(XX, YY, rho[0])
    ax.plot_wireframe(XX, YY, ZZ, rstride=rstride, cstride=cstride, label='%0.2f' % (rho[0],), color='black')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(30, 120)

    ax.legend()

    #--------------------------
    # 2nd plot
    #--------------------------
    index+=1
    ax = fig.add_subplot(index, projection='3d')
    ZZ=binorm(XX, YY, rho[1])
    ax.plot_wireframe(XX, YY, ZZ, rstride=rstride, cstride=cstride, label='%0.2f' % (rho[1],), color='magenta' )
    # rotate view
    ax.view_init(30, 120)
    ax.legend()


    #--------------------------
    # 3nd plot
    #--------------------------
    index+=1
    ax=fig.add_subplot(index, projection='3d')

    ZZ=binorm(XX, YY, rho[2])
    ax.plot_wireframe(XX, YY, ZZ, rstride=rstride, cstride=cstride, color='r', label="cor %0.2f" % (rho[2],))

    # rotate view
    ax.view_init(30, 120)
    ax.legend()

    #--------------------------
    # 4th plot
    #--------------------------
    index+=1
    ax=fig.add_subplot(index, projection='3d')
    ZZ=binorm(XX, YY, rho[3])
    ax.plot_wireframe(XX, YY, ZZ, rstride=rstride, cstride=cstride, color='green', label="cor %0.2f" % (rho[3],))
    # rotate view
    ax.view_init(30, 120)
    ax.legend()

    plt.savefig("test.jpg")






if __name__=='__main__':
    transform(rho=[-.0, -.3, -.6, -.9])