
import matplotlib
import matplotlib.pyplot as plt


def hello():
    matplotlib.use('Agg')
    fig, axes=plt.subplots()

    plt.plot([1, 2, 3, 4], 'ro')
    plt.axis([0, 6, 0, 20])



if __name__=='__main__':
    hello()