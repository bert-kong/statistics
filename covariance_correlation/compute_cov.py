import numpy as np
import matplotlib.pyplot as plt

"""
Calcium in Soil
table 3.3
page 71
Number y1 y2 y3
"""

X = np.array([\
    [35, 3.5 , 2.80],
    [35, 4.9 , 2.70],
    [40, 30.0, 4.38],
    [10, 2.8 , 3.21],
    [6 , 2.7 , 2.73],
    [20, 2.8 , 2.81],
    [35, 4.6 , 2.88],
    [35, 10.9, 2.90],
    [35, 8.0 , 3.28],
    [30, 1.6 , 3.20],
])

def test():
    n, _ = X.shape
    mean = X.mean(axis=0)
    Xadj = np.matrix(X - mean)

    SS = (Xadj.T * Xadj)/(n-1)
    print SS


if __name__ == '__main__':
    test()
