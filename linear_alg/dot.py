import numpy as np
import matplotlib.pyplot as plt


def test():
    p1 = np.array([4, 5])
    p2 = np.array([7, 2])

    unit1 = p1/np.linalg.norm(p1)
    unit2 = p2/np.linalg.norm(p2)

    t = np.linspace(0, 1.0, 7, endpoint=False)
    arr = np.zeros(shape=(7, 2))
    projected = np.zeros(shape=(7, 2))

    for i, ratio in enumerate(t):
        p = ratio * p1
        arr[i] = p
        # point p is projected p2
        projected[i] = unit2 * np.dot(p, unit2)
        i += 1


    x, y = p1
    plt.plot([0, x], [0, y], linestyle='-', color="green")

    x, y = p2
    plt.plot([0, x], [0, y], linestyle='-', color="green")
    plt.scatter(p1[0], p1[1], marker="o", color="black")
    plt.scatter(p2[0], p2[1], marker="o", color="black")

    plt.scatter(arr.T[0], arr.T[1], marker="o", color="green")
    plt.scatter(projected.T[0], projected.T[1], marker="o", color="red")

    for i in range(7):
        x1, y1 = arr[i]
        x2, y2 = projected[i]
        plt.plot([x1, x2], [y1, y2], linestyle='-.', color="blue")

    plt.grid(True)

    plt.savefig("test.jpg")
if __name__ == '__main__':
    test()