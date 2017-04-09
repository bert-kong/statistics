import numpy as np
import matplotlib.pyplot as plt

"""
Methods of Multivariate Analysis
-by Rencher

Page 289

"""

def test():
    X = np.array([ \
            [33, 36, 35, 38, 40],
            [60, 61, 64, 63, 65],
        ], dtype=np.float32)

    Y = np.array([ \
            [35, 36, 38, 39, 41, 43, 41],
            [57, 59, 59, 61, 63, 65, 59]
        ], dtype=np.float32)

    # compute means of data
    num_x = X.shape[1]
    num_y = Y.shape[1]
    mux = X.sum(axis=1)/num_x
    muy = Y.sum(axis=1)/num_y
    print "----- average of Y1 -------"
    print mux
    print "----- average of Y2 -------"
    print muy
    print "\n"

    # 0 means
    Xmu0 = X - mux.reshape(2, 1)
    Ymu0 = Y - muy.reshape(2, 1)

    #---------------------------------
    #  Assume : variances are the same
    #  concatenate 2 data to 1
    #---------------------------------
    Z = np.concatenate([Xmu0.transpose(),
                       Ymu0.transpose()])
    Z = Z.transpose()

    SS = (Z * Z).sum(axis=1)/(num_x + num_y-2)

    SSxy = (Z[0] * Z[1]).sum()/(num_x + num_y -2)
    D = np.diag(SS)
    SSxy = np.matrix([[0, SSxy],[SSxy, 0]])
    SSpl = D + SSxy
    print SSpl

    m_xy = mux - muy
    print m_xy
    a = SSpl.I * m_xy.reshape(2, 1)
    a = a.ravel()
    print "vector maximize the variance(estimate) ===> ", a

    U = np.dot(a, X)
    V = np.dot(a, Y)
    print np.floor(U * 100)/100
    print np.floor(V * 100)/100

if __name__ == '__main__':
    test()