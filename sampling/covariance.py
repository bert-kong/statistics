import numpy as np
import matplotlib.pyplot as plt


def test(degree=65, scale=(2., 0.3), num=100):

    radian = np.deg2rad(degree)
    sin = np.sin(radian)
    cos = np.cos(radian)

    R = np.matrix([[cos, -sin],
                   [sin, cos]], dtype=np.float32)

    S = np.diag(scale)
    cov = R * S * R.transpose()
    X = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=num).transpose()

    mean = X.sum(axis=1)/float(num)

    print "-------- mean -----------"
    print mean


    print "\n"
    print "-------- COV -----------"
    print cov

    #Y = X - mean.reshape(2, 1)
    Y = X
    SS = (Y ** 2).sum(axis=1)/(num-1)

    Z = Y[0] * Y[1]
    SSxy = Z.sum()/(num-1)
    A = np.array([[0, SSxy],
                  [SSxy, 0]])
    D = np.diag(SS)

    print "\n"
    print "-------- SS/n-1 COV from sampling data ------------"
    print D + A





if __name__ == '__main__':
    test(num=100000)