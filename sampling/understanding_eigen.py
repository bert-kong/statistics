import numpy as np
import matplotlib.pyplot as ply


def coord_attributes(ax, a=1., b=1.):

    ax.set_xlim(-a, b)
    ax.set_ylim(-a, b)
    ax.grid("on")
    ax.set_aspect("equal", adjustable="box")


def get_matrix(degree, scale=[1, 1]):

    rad = np.radians(degree)
    sin = np.sin(rad)
    cos = np.cos(rad)

    R = np.matrix([[cos, -sin], [sin, cos]])
    S = np.diag(scale)
    return (R * S)


def original_dataset():
    #---------------------------------
    #  circle point set
    #---------------------------------
    nr_circle_points = 50

    X = np.linspace(start=-1.0,
                    stop=1.0,
                    num=nr_circle_points,
                    endpoint=True)

    Y = np.sqrt(1-(X ** 2))


    #---------------------------------
    #  line Y date set
    #---------------------------------
    nr_line_points = 40
    X1 = np.linspace(start=-1,
                     stop=1,
                     num=nr_line_points)

    Y1 = np.zeros(shape=(nr_line_points, ))

    #---------------------------------
    #  line x = 0
    #---------------------------------
    #X2 = np.zeros(shape=(100, ))
    #Y2 = np.linspace(start=-1.0, stop=1.0, num=100)


    # all points circle + line points
    # original data set

    XX = np.concatenate([X,  X, X1], axis=0)
    YY = np.concatenate([Y, -Y, Y1], axis=0)

    # jet, rainbow, hsv, prism, inferno
    colormap = ply.get_cmap("hot")
    #colors = colormap(np.r_[XX, YY])
    colors = colormap(XX)


    return XX, YY, colors


def transform(scale=[1, 1]):

    X, Y, colors = original_dataset()

    M = get_matrix(degree=45, scale=scale)
    XY = np.matrix([X, Y])
    data = M * XY

    return np.array(data[0, :]), np.array(data[1, :]), colors


def get_matrix_eigen(degree=45, scale=[1, 1]):

    rad = np.radians(degree)
    sin = np.sin(rad)
    cos = np.cos(rad)

    R = np.matrix([[cos, -sin], [sin, cos]])
    S = np.diag(scale)

    return R * S * R.I


def transform_eigen(scale=[1, 1]):

    X, Y, colors = original_dataset()

    M = get_matrix_eigen(degree=45, scale=scale)
    XY = np.matrix([X, Y])
    data = M * XY

    return np.array(data[0, :]), np.array(data[1, :]), colors


def eigen_vector():
    # graph
    fig = ply.figure()

    a, b = 3.0, 3.0
    #---------------------------------
    #   normal scale and rotation
    #---------------------------------
    ax = fig.add_subplot(121)
    coord_attributes(ax, a=a, b=a)

    #------------------------------
    X, Y, colors = original_dataset()
    ax.scatter(X, Y, color=colors, marker='o', s=10.0)


    scale = [3, 1.]

    # scale and rotation
    X, Y, _ = transform(scale)
    ax.scatter(X, Y, color=colors, marker='o', s=10.0)


    #---------------------------------
    #   eigen scale and rotation
    #---------------------------------
    ax2 = fig.add_subplot(122)
    coord_attributes(ax2, a=a, b=b)

    X, Y, colors = original_dataset()
    ax2.scatter(X, Y, color=colors, marker='o', s=10.0)

    X, Y, _ = transform_eigen(scale)
    ax2.scatter(X, Y, color=colors, marker='o', s=10.0)


    ply.savefig("test.jpg")



if __name__=="__main__":
    eigen_vector()
