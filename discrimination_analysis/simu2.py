import numpy as np
import matplotlib.pyplot as plt


class SimuDis(object):
    """
    generate 2 groups of bivariable
    normal distribution data

    """

    def __init__(self, num=15, degree=65, scale=(2, .3)):

        self._data = None
        self._data_transformed = None
        self._max_vector = None

        #-----------------------
        self._bound = 4

        self._num = num
        self._radian=np.deg2rad(degree)
        scale = np.array(scale)
        self._S = np.diag(scale)

        sin = np.sin(self._radian)
        cos = np.cos(self._radian)
        self._R = np.matrix([[cos, -sin],
                             [sin, cos]])

        #-------------------------------------
        #  Covariance matrix and mean vector
        #-------------------------------------
        self._COV = self._R * self._S * self._R.transpose()
        self._mean = np.array([[1., 2.], [4., 1.5]])


        # generate bivariate normal distribution
        #self.data_normal()

        # plot object
        self._fig = plt.figure()


    def data_circle(self):
        """
        perfect circle X, Y
        :return:
        """

        X = np.linspace(start=-1, stop=1, num=self._num, endpoint=True)
        Y = np.sqrt(1 - (X * X))

        X = np.concatenate([X, X], axis=0)
        Y = np.concatenate([Y, -Y], axis=0)

        self._data = np.array([X, Y])


    def data_normal(self):
        """
        standardized multivariate normal data
        :return:
        """

        I = np.matrix([[1., 0],
                       [0, 1.]])

        mean = np.array([0., 0.])

        X = np.random.multivariate_normal(mean=mean,
                                          cov=I,
                                          size=self._num)
        self._data = X.T

    def bivariate_normal(self):
        self._data_transformed = []

        for mean in self._mean:
            X = np.random.multivariate_normal(mean=mean,
                                              cov=self._COV,
                                              size=self._num)
            X = X.transpose()
            self._data_transformed.append(X)



    def transformation(self):


        self._data_transformed = []
        for p in self._mean:
            Y = np.array(self._COV * self._data + p.reshape(2, 1))
            self._data_transformed.append(Y)


    def _set_plot_attribute(self, plot):
        b = self._bound

        plot.grid("on")
        plot.set_xlim(-b, b)
        plot.set_ylim(-b, b)
        plot.axhline(y=0, linewidth=2, color="blue")
        plot.axvline(x=0, linewidth=2, color="blue")
        plot.set_aspect("equal", adjustable="box")

    def discriminant_analysis(self):

        X, Y = self._data_transformed

        #---------------------------------
        # 0 mean data
        #---------------------------------
        mu0, mu1 = self._mean
        print "mean 1 (sampling) ---> ",  X.sum(axis=1)/self._num
        print "mean 1 ---->", mu0
        print "mean 2 (sampling)  ---> ", Y.sum(axis=1)/self._num
        print "mean 2 ---->", mu1


        #---------------------------------
        # Covariance
        #---------------------------------

        mu0.shape = (2, 1)
        mu1.shape = (2, 1)

        # 0 means
        Xm0 = X - mu0
        Ym0 = Y - mu1

        # X, Y have the same variacne
        # combine the Xm0 and Ym0
        Z = np.concatenate([Xm0.T, Ym0.T]).T

        SS = ((Z ** 2).sum(axis=1))/(2 * self._num-1)

        SSxy = (Z[0] * Z[1]).sum()/(2 * self._num-1)


        D = np.diag(SS)
        M = np.matrix([[0, SSxy],
                       [SSxy, 0]])

        SS = D + M
        print "-------------- SS/COV from sampling data ------"
        print SS

        print "\n"
        print "-------------- COV  ------"
        print self._COV

        print "\n"
        print "---------  Roation Matrix ---------"
        print self._R

        print "\n"
        print "---------  Scaler Matrix ---------"
        print self._S

        print "\n"
        print "---------  Eigen Values/Vectors ---------"
        ei_val, ei_vec = np.linalg.eig(self._COV)
        print ei_val
        print ei_vec

        mu = self._mean[0] - self._mean[1]
        a = SS.I * mu.reshape(2, 1)

        self._max_vector = np.array(a)
        print "\n"
        print "--------- Maximun Vector ---------"
        print self._max_vector


    def draw(self):
        ax = self._fig.add_subplot(111)

        X, Y = self._data_transformed
        ax.scatter(X[0, :], X[1, :], marker='.', color='magenta')
        ax.scatter(Y[0, :], Y[1, :], marker='o', color='red')

        #----------------------------
        #   mean points
        #----------------------------
        p1, p2 = self._mean
        ax.scatter(p1[0], p1[1], marker='.', color='black')
        ax.scatter(p2[0], p2[1], marker='.', color='black')

        #----------------------------
        #   mean points
        #----------------------------
        vec_x = np.array([self._max_vector[0], -self._max_vector[0]])
        vec_y = np.array([self._max_vector[1], -self._max_vector[1]])
        ax.plot(vec_x, vec_y, linestyle='-', color='black')


        #---------------------------
        #  set plot attributes
        #---------------------------
        self._set_plot_attribute(ax)

        plt.savefig("test.jpg")

    def _set_xylim(self, b):
        self._bound = b

    def _get_xylim(self):
        return self._bound


    xylim=property(fget=_get_xylim, fset=_set_xylim)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean


    def info(self):
        X, Y = self._data_transformed
        print "---- transformed data shape ---- "
        print X.shape
        print Y.shape


def test():
    sim = SimuDis(num=1000)
    sim.bivariate_normal()
    sim.info()
    sim.discriminant_analysis()
    sim.xylim = 8
    sim.draw()




if __name__ == '__main__':
    test()