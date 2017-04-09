import numpy as np
import matplotlib.pyplot as plt


def logistic(B0, B1, X):
    # ln(1/(1-p) = B0 + B1X
    # 1/(1-p) = exp(B0 + B1X)
    # p = exp(B0 + B1X)/[1 + exp(B0 + B1X)]

    # P probabilities : P ~ B0 + B1
    exp = np.exp(B0 + (B1 * X))
    P = exp/(1 + exp)

    return P


def draw_logistic():

    #-----------------------------------
    a = 5
    X = np.arange(-a, a, .01)

    #-----------------------------------
    fig = plt.figure()
    index = 111
    ax = fig.add_subplot(index)

    B0, B1 = 1, 2
    P = logistic(B0, B1, X)
    ax.plot(X, P, color='g', linestyle='-')

    B0, B1 = 2, 2
    P = logistic(B0, B1, X)
    ax.plot(X, P, color='b', linestyle='-')

    B0, B1 = 3, 2
    P = logistic(B0, B1, X)
    ax.plot(X, P, color='y', linestyle='-')

    B0, B1 = 1, .5
    P = logistic(B0, B1, X)
    ax.plot(X, P, color='c', linestyle='-')

    B0, B1 = 1, 1.5
    P = logistic(B0, B1, X)
    ax.plot(X, P, color='m', linestyle='-')

    B0, B1 = 1, 2.5
    P = logistic(B0, B1, X)
    ax.plot(X, P, color='b', linestyle='-')

    plt.grid(True)
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.axhline(y=.5)
    plt.xlim([-a, a])
    #plt.ylim([-.2, 1.2])
    plt.ylim([-.1, 1.1])
    plt.savefig('test.jpg')


def draw(B0, B1, ax):
    """
    generate P and Y data
    :param B0:
    :param B1:
    :param ax:
    :return:
    """

    #------------------------------------
    # generate P and Y data
    #------------------------------------

    #------------------------------------
    #
    # P = f(X) or P = B0 + B1X + B2X etc
    # ln(P/(1+P)) = B0 + B1X  P is Probabilities ~ (0, 1)
    # Y ~ Binomial(1, P) / Bernouli
    #------------------------------------
    bound = 5
    X = np.arange(-bound, bound, .1)

    P = logistic(B0, B1, X)

    # initial Y to 0s
    Y = np.zeros(shape=X.shape, dtype=X.dtype)

    i = 0
    for p in P:
        Y[i] = np.random.binomial(1, p, 1)
        i += 1

    ax.plot(X, P, color='b', linestyle='-')
    ax.scatter(X, Y, color='g', marker='o')
    ax.axvline(x=0, linewidth=2, color='green')
    ax.axhline(y=0, linewidth=2, color='green')
    ax.axhline(y=.5, linewidth=1, color='r')
    ax.set_title("B0 %0.2f  B1 %0.2f" % (B0, B1))
    #ax.legend()
    ax.grid('on')



def compare():
    fig = plt.figure()

    index = 221
    ax = fig.add_subplot(index)
    B0, B1 = 0, 2
    draw(B0, B1, ax)

    index+=1
    ax = fig.add_subplot(index)
    B0, B1 = 3, 2
    draw(B0, B1, ax)

    index+=1
    ax = fig.add_subplot(index)
    B0, B1 = -3, 2
    draw(B0, B1, ax)

    index+=1
    ax = fig.add_subplot(index)
    B0, B1 = 0, 0.4
    draw(B0, B1, ax)

    plt.savefig('test.jpg')



if __name__ == '__main__':

    compare()