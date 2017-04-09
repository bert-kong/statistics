import numpy as np
import matplotlib.pyplot as plt


def draw():

    X = np.arange(start=0.0001, stop=1, step=.001)

    odds = lambda P : P/(1-P)

    Y = odds(X)

    plt.plot(X, Y, color='g', linestyle='-')
    plt.plot(X, np.log(Y), color='r', linestyle='-')

    plt.grid(True)
    plt.xlim([0, 1.2])
    plt.ylim([-10, 10])
    plt.axvline(x=.5)
    plt.savefig("test.jpg")

if __name__ == '__main__':
    draw()