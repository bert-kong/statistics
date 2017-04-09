import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def test(degree=70):

    rad = np.radians(degree)
    cos = np.cos(rad)
    sin = np.sin(rad)

    # new coordinates or projection vectors
    V1 = np.array([cos, -sin])
    V2 = np.array([sin, cos])

    #----------------------------
    #  pointers for a line
    #  (0, 0) - (4, 2)
    #----------------------------
    size = 5
    P = np.array([4, 2])
    direction = np.linalg.norm(P)
    t = np.linspace(0, 1, size, endpoint=True)
    L = np.vstack([scalar * P for scalar in t])

    Y1 = np.dot(V1, L.T)
    Y2 = np.dot(V2, L.T)

    #----------------------------
    # plotting
    #----------------------------
    colors = cm.inferno(t)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y = L.T[0], L.T[1]
    ax.scatter(X, Y, marker='o', color="magenta")

    #----------------------------
    #  plot new coordinates
    #----------------------------
    U1 = np.vstack((-V1, V1))
    U2 = np.vstack((-V2, V2))
    ax.plot(U1.T[0], U1.T[1], linestyle="-", color="blue")
    ax.plot(U2.T[0], U2.T[1], linestyle="-", color="green")

    #----------------------------
    #  plot Y1, Y2
    #----------------------------
    ax.scatter(Y1, Y2, marker="o", color='red')

    #----------------------------
    #  projections on V1, V2
    #----------------------------
    Z1 = np.vstack([x * V1 for x in Y1])
    Z2 = np.vstack([x * V2 for x in Y2])
    ax.scatter(Z1.T[0], Z1.T[1], marker=".", s=4, color='black')
    ax.scatter(Z2.T[0], Z2.T[1], marker=".", s=4, color='black')

    #----------------------------
    # set attributes
    #----------------------------
    ax.grid("on")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(y=0)
    ax.axvline(x=0)
    ax.set_aspect("equal", adjustable="box")

    plt.savefig("test.jpg")


if __name__ == '__main__':
    test(degree=40)