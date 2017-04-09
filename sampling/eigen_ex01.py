import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Eigen(object):

    def __init__(self, degree=40, scale=[4, 2]):

        self.size = 6
        self.b = 4
        radian = np.radians(degree)
        S = np.diag(scale)
        cos = np.cos(radian)
        sin = np.sin(radian)

        self.e1 = np.array([cos, sin])
        self.e2 = np.array([-sin, cos])

        R = np.matrix([cos, -sin, sin, cos]).reshape(2, 2)
        COV = R * S * R.T

        self.S = S
        self.R = R
        self.COV = COV


    def bivariate(self):

        pass


    def circle(self):

        X1 = np.linspace(-1, 1, num=self.size, endpoint=True)
        X2 = np.sqrt(1-(X1*X1))

        X1 = np.hstack((X1, X1))
        X2 = np.hstack((X2, -X2))

        self.X = np.vstack([X1, X2])

    def set_plot_attr(self, ax):
        b = self.b

        ax.grid("on")
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)
        ax.axhline(y=0, color="blue")
        ax.axvline(x=0, color="blue")
        ax.set_aspect("equal", adjustable="box")

    def draw_eigen_vector(self, ax):
        e1, e2 = self.e1, self.e2
        E = np.vstack((e1, -e1))
        Y1 = 10 * E.T[0]
        Y2 = 10 * E.T[1]

        ax.plot(Y1, Y2, linestyle="-", color="green")

        E = np.vstack((e2, -e2))
        Y1 = 10 * E.T[0]
        Y2 = 10 * E.T[1]
        ax.plot(Y1, Y2, linestyle="-", color="green")


    def draw(self):

        # plot
        fig = plt.figure()
        index = 211
        ax = fig.add_subplot(index)

        # preparing data
        X1, X2 = self.X[0], self.X[1]

        #ax.scatter(X1, X2, c=range(self.size * 2), marker="o", cmap=cm.get_cmap("RdYlBu"))
        color_map = cm.get_cmap("RdBu")

        colors = range(self.size * 2)
        ax.scatter(X1, X2, c=colors, marker="o", s=20, cmap=color_map)

        #-----------------------------------
        #   "linear" mapping
        #-----------------------------------
        Y = self.COV * self.X
        ax.scatter(Y[0], Y[1], c=colors, marker="o", s=20, cmap=color_map)
        self.draw_eigen_vector(ax)

        self.set_plot_attr(ax)

        #-----------------------------------
        #   "non linear mapping
        #-----------------------------------
        S, R = self.S, self.R
        Y = (R * S) * self.X
        index += 1
        ax2 = fig.add_subplot(index)
        ax2.scatter(X1, X2, c=colors, marker="o", s=20, cmap=color_map)
        ax2.scatter(Y[0], Y[1], c=colors, marker="o", s=20, cmap=color_map)
        self.draw_eigen_vector(ax2)

        self.set_plot_attr(ax2)

        # save image to test.jpg
        plt.savefig("test.jpg")


if __name__ == '__main__':
    eig = Eigen()
    eig.circle()
    eig.draw()