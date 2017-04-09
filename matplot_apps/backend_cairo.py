
import matplotlib.backends.backend_cairo as bk_cairo
import matplotlib.pyplot as plt
import numpy as np



def test():

    f=lambda t : np.exp(-t) * np.cos(2 * np.pi * t)

    t1=np.arange(start=0.0, stop=5.0, step=.1)
    t2=np.arange(start=0.0, stop=5.0, step=.02)

    fig=plt.figure(1)
    ax = plt.subplot(211)

    plt.plot(t1, f(t1), 'bo')
    plt.plot(t2, f(t2), 'k')

    ax = plt.subplot(212)

    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')

    # set cairo canvas/surface for backend draw
    canvas=bk_cairo.FigureCanvasCairo(fig)

    plt.savefig("/home/projects/py/statistics/matplot_apps/test01.jpg")


if __name__=='__main__':
    test()


