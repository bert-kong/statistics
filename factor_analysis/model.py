import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
   Example: 2-factor model of 3 variables (m=2, p=3)

   underlying latent common factors: m=2 F1, F2
   specific errors : e1, e2, e3
   observed variables: p=3  X1, X2, X3
   loading/coefficients: L 2x3 is (pxm) factor loading matrix
   m = 2 is common factors, the ei are the p=3 specific errors
   l(ij) pxm factor loadings
       (l11, l12,
        l21, l22,
        l31, l32)

   X = L * F + e

   VAR(X) = L * t(L) + VAR(e)


   Assumptions:
       F mean (0, 0)
         variance (1, 0
                   0, 1)
       F independent
"""

def circle(num=100):
    X = np.linspace(-1, 1, num=num)
    Y = np.sqrt(1-(X**2))

    return (X, Y)


def transformation(F1, F2):

    I = np.diag([1, 1])

    L = np.matrix(np.random.randint(low=1, high=20, size=(3, 2)))

    # variance
    VAR = L * I * L.transpose()

    print "------------- Factor/Loading Matrix --------------"
    print L
    print "------------- Variance/Covariance --------------"
    print VAR

    print "------------- Rotation and Scalar --------------"
    print np.linalg.eig(VAR)

    F = np.matrix([F1, F2])
    X = L * F

    return (np.array(X[0])[0],
            np.array(X[1])[0],
            np.array(X[2])[0])


def set_plot_attr(plot):
    plot.grid("on")

    b = 5
    plot.set_xlim(-b, b)
    plot.set_ylim(-b, b)
    plot.set_aspect("equal", adjustable="box")
    plot.axhline(y=0, color="blue", linewidth="1")
    plot.axvline(x=0, color="blue", linewidth="1")




def test():
    fig = plt.figure()
    index = 111
    ax = fig.add_subplot(index, projection='3d')

    # latent/underlying factors
    F1, F2 = circle(100)
    ax.plot(F1, F2, color="green", linestyle='-')
    set_plot_attr(ax)

    #------------------------------------
    X1, X2, X3 = transformation(F1, F2)

    plt.savefig("test.jpg")





if __name__ == '__main__':
    test()