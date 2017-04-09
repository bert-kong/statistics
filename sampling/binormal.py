import numpy as np
import matplotlib.pyplot as plt



class BivariateNormal(object):

    def __init__(self,
                 degree=67,
                 scale=[3, .2],
                 mean=np.array([[3, 1], [-1, 6]]),
                 size=1000):

        self.size = size
        radian = np.radians(degree)
        S = np.diag(scale)
        cos = np.cos(radian)
        sin = np.sin(radian)

        V1 = np.array([cos, -sin])
        V2 = np.array([sin, cos])
        self.V1 = V1
        self.V2 = V2

        R = np.matrix([V1, V2])

        self.COV = R * S * R.T
        self.R = R
        self.S = S

        self.mean = mean

        mu0 = mean[0]
        mu1 = mean[1]

        data = np.random.multivariate_normal(mean=[0, 0],
                                             cov=self.COV,
                                             size=size)

        self.data = data

        X1= np.random.multivariate_normal(mean=mu0,
                                          cov=self.COV,
                                          size=size)

        X2 = np.random.multivariate_normal(mean=mu1,
                                           cov=self.COV,
                                           size=size)

        self.X1 = X1
        self.X2 = X2
        self.X = np.vstack((X1, X2))

        self.ss()


    def ss(self):

        X1, X2 = self.X1, self.X2
        mu1, mu2 = self.mean[0], self.mean[1]

        Z1 = X1.T - mu1.reshape(2,1)
        Z2 = X2.T - mu2.reshape(2,1)

        Z = np.hstack((Z1, Z2))

        size = 2 * self.size
        SS = (Z ** 2).sum(axis=1)/(size-1)
        D = np.diag(SS)

        SSxy = (Z[0] * Z[1]).sum()/(size-1)
        A = np.matrix([[0, SSxy],
                       [SSxy, 0]])

        self.SS = D + A


    def verify(self):
        """
        verify max direction of variance
        :return:
        """

        pass

    def rotation_line(self):

        # horizontal line
        nr_points = 10
        X1 = np.linspace(-10, 10, nr_points)
        X2 = np.zeros(shape=(nr_points,))
        L = np.vstack((X1, X2))
        Y1 = np.dot(self.V1, L)
        Y2 = np.dot(self.V2, L)
        print "-----"
        print Y1
        print Y2

        x, y =  Y1[-1], Y2[-1]
        print "(x, y) ", x, y
        Z1 = np.array([y, -x])
        Z = np.vstack((Z1, -Z1))

        self.ax.plot(Y1, Y2, linestyle="-", color="#F00000", linewidth=2)
        self.ax.plot(Z.T[0], Z.T[1], linestyle="-", color="#F00000", linewidth=2)


    def discriminant_line(self):
        """
        direction max mean delta/variance
        :return:
        """

        mu1 = self.mean[0]
        mu2 = self.mean[1]

        mean = mu2-mu1

        D = (self.COV.I * mean.reshape(2,1)).ravel()
        self.components = D

        Z = np.vstack((-D, D))
        Z1 = Z[:, 0]
        Z2 = Z[:, 1]
        self.ax.plot(Z1, Z2, linestyle="-", color="black")


    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.ax = ax

        data = self.data
        ax.scatter(data.T[0], data.T[1], marker="o", s=2, color="yellow")

        X1 = self.X1
        X2 = self.X2

        b = 10
        ax.scatter(X1.T[0], X1.T[1], marker="o", s=2, color="red")
        ax.scatter(X2.T[0], X2.T[1], marker="o", s=2, color="magenta")

        self.rotation_line()
        self.discriminant_line()

        ax.grid(True)
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)
        ax.axhline(y=0, linewidth=1, color="blue")
        ax.axvline(x=0, linewidth=1, color="blue")
        ax.set_aspect("equal", adjustable="box")

        plt.savefig("test.jpg")



if __name__ == '__main__':

    norm = BivariateNormal(size=2000)
    norm.draw()
    print norm.SS
    print "\n"
    print norm.COV
