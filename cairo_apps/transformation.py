import cairo
import numpy as np
import StringIO
from PIL import Image
import os

"""
matrix transform

x' = x + y
y' = x - y

A = (1,  1,
     1, -1)
"""

cur_dir='/home/projects/py/statistics/cairo_apps'
im_dir='/home/projects/Pictures'
def transformation():

    buffer=StringIO.StringIO()
    im=Image.open(os.path.join(im_dir, "small_house.jpeg"), mode='r')
    im.save(buffer, format='png')
    width, height = im.size
    im.close()


    # reset buffer pionter
    buffer.seek(0)

    #----------------------------------
    # source image
    #----------------------------------
    src_surf=cairo.ImageSurface.create_from_png(buffer)

    #----------------------------------
    # canvas/destination image
    # transformation : 1 <-> 1
    #    x' = 1x + 3y + a
    #    y' = 2x - 4y + b
    #    cairo.Matrix(1, 2, 3, -4, a, b)
    #----------------------------------

    w = width + height
    h = width + height
    ar=np.zeros(shape=(h, w, 4), dtype=np.uint8)
    canvas=cairo.ImageSurface.create_for_data(ar,
                                               cairo.FORMAT_ARGB32,
                                               w,
                                               h,
                                               w * 4)

    # create a context for storing drawing states
    ctx = cairo.Context(canvas)
    m=cairo.Matrix(1, 1, 1, 0, 0, h/2)
    # load matrix into context

    # save states
    ctx.save()
    ctx.transform(m)
    ctx.scale(.5, .5)

    # load the source image
    ctx.set_source_surface(src_surf)

    # draw on canvas
    ctx.rectangle(0, 0, w, h)
    ctx.fill()
    ctx.restore()

    ctx.translate(w/2, h/4)
    #ctx.scale(0.5, 0.5)

    ctx.set_source_surface(src_surf)
    ctx.rectangle(0, 0, width, height)
    ctx.fill()

    im=Image.fromarray(ar, mode='RGBA')
    im.save(os.path.join(cur_dir, 'test1.jpg'))

if __name__=='__main__':
    transformation()





