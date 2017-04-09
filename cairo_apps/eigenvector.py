
import numpy as np
import cairo
import os
import StringIO
from PIL import Image

"""
A = ( (3 1),
      (2 2))

      x = 3x + y
      y = 2x + 2y
"""

im_dir="/home/dicaprio/Pictures"

cur_dir=os.path.dirname(__file__)

def compute_eigenvalues():

    m = np.matrix([[3, 1], [2, 2]], dtype=np.int32)
    eigen_value, eigen_vector=np.linalg.eig(m)
    print eigen_value
    print eigen_vector


def image_transform():

    #---------------------------
    # source image
    #---------------------------
    im=Image.open(os.path.join(im_dir, "building.jpeg"), mode='r')

    buffer=StringIO.StringIO()
    im.save(buffer, format='png')
    buffer.seek(0)

    src_surf = cairo.ImageSurface.create_from_png(buffer)
    w, h=im.size

    #---------------------------
    # canvas
    #---------------------------
    x, y = w, h
    width=  (3 * x) + (1 * y)
    height= (2 * x) + (2 * y)

    ar = np.zeros(shape=(height, width, 4), dtype=np.uint8)
    canvas=cairo.ImageSurface.create_for_data(ar,
                                              cairo.FORMAT_ARGB32,
                                              width,
                                              height,
                                              ar.strides[0])

    #---------------------------
    # Context: matrix etc
    #---------------------------
    ctx=cairo.Context(canvas)

    m=cairo.Matrix(3, 2, 1, 2, 0, 0)

    # adding matrix to context
    ctx.transform(m)

    # adding the source image to context
    ctx.set_source_surface(src_surf)

    # create rectangle in the context
    ctx.rectangle(0, 0, width, height)

    # draw on canvas/numpy array
    ctx.fill()

    # save and verify the image
    im=Image.fromarray(ar, mode='RGBA')
    im.save(os.path.join(cur_dir, "test.jpg"))


if __name__=='__main__':
    compute_eigenvalues()
    image_transform()


