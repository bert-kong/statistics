import numpy as np
import matplotlib.pyplot as plt

"""
http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc552.htm
"""
def test():
    X1 = np.array([7., 4., 6, 8, 8, 7, 5, 9, 7, 8])
    X2 = np.array([4., 1., 3, 6, 5, 2, 3, 5, 4, 2])
    X3 = np.array([3., 8, 5, 1, 7, 9, 3, 8, 5, 2])

    X = np.vstack((X1, X2, X3))

    cor = np.corrcoef(X)
    print " corr coef"
    print cor
    _, n = X.shape
    mean = X.mean(axis=1)

    Xadj = X - mean.reshape(3, 1)
    SS = ((Xadj ** 2).sum(axis=1))/(n-1)
    D = np.diag(SS)

    #---------------------------------------
    #   Variance-Covariance Matrix
    #---------------------------------------
    SS01 = np.sum(Xadj[0] * Xadj[1])/(n-1)
    SS02 = np.sum(Xadj[0] * Xadj[2])/(n-1)
    SS12 = np.sum(Xadj[1] * Xadj[2])/(n-1)
    Cov = np.matrix([[0, SS01, SS02],
                     [SS01, 0, SS12],
                     [SS02, SS12, 0]])

    COV = D + Cov

    values, vectors = np.linalg.eig(COV)
    values_norm = values/np.linalg.norm(values)

    print mean
    print COV
    print "-----------Eigen values----------"
    print values
    print values_norm
    print "\n"
    print "-----------Eigen vectors----------"
    print vectors

    #---------------------------------------
    #   Variance-Covariance Matrix
    #---------------------------------------
    SS0 = np.sqrt(np.sum(Xadj[0] * Xadj[0]))
    SS1 = np.sqrt(np.sum(Xadj[1] * Xadj[1]))
    SS2 = np.sqrt(np.sum(Xadj[2] * Xadj[2]))

    SS01 = np.sum(Xadj[0] * Xadj[1])/(SS0 * SS1)
    SS02 = np.sum(Xadj[0] * Xadj[2])/(SS0 * SS2)
    SS12 = np.sum(Xadj[1] * Xadj[2])/(SS1 * SS2)

    cor = np.matrix([[1, SS01, SS02],
                     [SS01, 1, SS12],
                     [SS02, SS12, 1]])

    values, vectors = np.linalg.eig(cor)
    print "\n"
    print "----------- Correlation ----------"
    print cor
    print "\n"

    print "-----------Eigen values----------"
    print values
    print "\n"
    print "-----------Eigen vectors----------"
    print vectors





if __name__ == '__main__':
    test()