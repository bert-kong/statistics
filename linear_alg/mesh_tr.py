import numpy as np
import matplotlib.pyplot as plt


def test():
    X = np.array([-2, -1, 0, 1, 2])
    Y = np.array([-2, -1, 0, 1, 2])

    XX, YY = np.meshgrid(X, Y)

    VV = np.zeros(shape=(5, 5))
    UU = np.zeros(shape=(5, 5))

    A1 = np.random.randint(low=1, high=5, size=2)
    coef = np.random.random() * np.random.randint(7, 14)
    A2 = coef * A1
    A = np.matrix([A1, A2])
    print A

    for i in range(5):
        Vec = np.array([XX[i], YY[i]])
        TT = A * Vec
        VV[i] = TT[0]
        UU[i] = TT[1]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(XX, YY, color="green", marker="o")
    ax.set_alpha(.3)
    ax.scatter(VV, UU, color="red", marker="o")

    ax.grid("on")
    b = 20
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.axhline(y=0, linewidth=1, color="blue")
    ax.axvline(x=0, linewidth=1, color="blue")

    plt.savefig("test.jpg")


if __name__ == '__main__':
    test()