import numpy as np
import matplotlib.pyplot as plt

"""
Methods of Multivariate Analysis 2nd
page 94
table 3-5
measurements on 1st, 2nd sons of 25 families
1st son
head length, head breadth
2nd son
head length, head breadth
"""
X = np.array([ \
    [191, 155, 179, 145],
    [195, 149, 201, 152],
    [181, 148, 185, 149],
    [183, 153, 188, 149],
    [176, 144, 171, 142],
    [208, 157, 192, 152],
    [189, 150, 190, 149],
    [197, 159, 189, 152],
    [188, 152, 197, 159],
    [192, 150, 187, 151],
    [179, 158, 186, 148],
    [183, 147, 174, 147],
    [174, 150, 185, 152],
    [190, 159, 195, 157],
    [188, 151, 187, 158],
    [163, 137, 161, 130],
    [195, 155, 183, 158],
    [186, 153, 173, 148],
    [181, 145, 182, 146],
    [175, 140, 165, 137],
    [192, 154, 185, 152],
    [174, 143, 178, 147],
    [176, 139, 176, 143],
    [197, 167, 200, 158],
    [190, 163, 187, 150],
], dtype=np.float64)

def compute_ss():

    n, _ = X.shape
    X1 = X[:, 0]
    X2 = X[:, 1]

    Y = np.vstack((X1, X2))
    mean = Y.mean(axis=1).reshape(2, 1)
    #---------------------------
    #  1st way
    #---------------------------
    Yadj = np.matrix(Y - mean)

    SS = (Yadj * Yadj.T)/(n-1)
    print SS

    #---------------------------
    #  2nd way
    #---------------------------
    Z = np.matrix(Y)
    SSunad = (Z * Z.T)/(n-1)
    M = np.matrix(mean)
    M = M * M.T
    print M
    SS = SSunad - M
    print SS


def test():
    n, _ = X.shape
    mean = X.mean(axis=0)
    print mean

    X0 = X[:, 0]
    X1 = X[:, 1]
    X0adj = X0.T - mean[0]
    X1adj = X1.T - mean[1]

    SS0 = np.dot(X0adj, X0adj)/(n-1)
    SS1 = np.dot(X1adj, X1adj)/(n-1)
    SS01 = np.dot(X0adj, X1adj)/(n-1)

    #------------------------------------------
    # sample variance and covariance from data
    #------------------------------------------
    SS = np.matrix([SS0, SS01, SS01, SS1]).reshape(2, 2)
    eig_val, eig_vec = np.linalg.eig(SS)
    #print eig_val
    print "\n"
    a1 = np.array(eig_vec[:, 0]).ravel()
    a2 = np.array(eig_vec[:, 1]).ravel()
    print "-----------------"
    print "   eigen vectors"
    print a1
    print a2
    print "-----------------"
    print "\n"

    Y = np.vstack((X0adj, X1adj))
    Dy = np.linalg.norm(Y, axis=0)
    print Dy
    Z1 = np.dot(a1, Y)
    Z2 = np.dot(a2, Y)
    Z = np.vstack((Z1, Z2))
    Dz = np.linalg.norm(Z, axis=0)
    print Dz




    e1 = np.vstack((a1, -a1)) * 50
    e2 = np.vstack((a2, -a2)) * 50

    print "---> ", mean
    mu = mean[[0, 1]].reshape(2, 1)
    print "---> ", mu
    e1 = e1.T + mu
    e2 = e2.T + mu

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X0, X1, marker="o", color="green")

    print "-----------------"
    print e1.T[0]
    print "-----------------"
    print e1.T[1]
    print "-----------------"
    ax.plot(e1[0], e1[1], linestyle="-", color="blue", linewidth=1)
    ax.plot(e2[0], e2[1], linestyle="-", color="blue", linewidth=1)

    x, y = mu
    ax.scatter([x], [y],  color="red", s=20)


    #-----------------------------------
    #  attributes
    #-----------------------------------
    ax.grid("on")
    b = 250
    ax.set_xlim(100, b)
    ax.set_ylim(100, b)
    ax.axhline(y=0, color="blue", linewidth=1)
    ax.axvline(x=0, color="blue", linewidth=1)
    ax.set_aspect("equal", adjustable="box")

    #-----------------------------------
    plt.savefig("test.jpg")


if __name__ == '__main__':
    #test()
    compute_ss()
