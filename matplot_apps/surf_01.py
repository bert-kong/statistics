import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def test():

    fig = plt.figure()


    # preparing data
    X=np.arange(-5, 5, .25)
    Y=np.arange(-5, 5, .25)

    XX, YY = np.meshgrid(X, Y)

    # (X, Y) -> R
    Z=np.sqrt(XX**2 + YY**2)

    # R -> R
    Z = np.sin(Z)

    index=221
    ax = fig.add_subplot(index, projection='3d')
    surf=ax.plot_surface(XX, YY, Z,
                         rstride=1,
                         cstride=1,
                         cmap=cm.coolwarm,
                         linewidth=0,
                         antialiased=False)

    ax.set_zlim(-1.01, 1.01)

    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    index+=1
    ax = fig.add_subplot(index, projection='3d')
    surf=ax.plot_surface(XX, YY, Z,
                         rstride=3,
                         cstride=3,
                         cmap=cm.coolwarm,
                         linewidth=1,
                         antialiased=False)

    ax.set_zlim(-1.01, 1.01)

    index+=1
    ax = fig.add_subplot(index, projection='3d')
    surf=ax.plot_surface(XX, YY, Z,
                         rstride=1,
                         cstride=1,
                         cmap=cm.coolwarm,
                         linewidth=0,
                         antialiased=False)
    ax.set_zlim(-1.01, 1.01)

    index+=1
    ax = fig.add_subplot(index, projection='3d')
    surf=ax.plot_surface(XX, YY, Z,
                         rstride=1,
                         cstride=1,
                         cmap=cm.coolwarm,
                         linewidth=0,
                         antialiased=False)

    ax.set_zlim(-1.01, 1.01)



    fig.colorbar(surf, shrink=.5, aspect=5)

    fig.savefig("test.jpg")



if __name__ == '__main__':
    test()