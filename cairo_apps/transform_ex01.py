
import cairo
import numpy as np
import os
from PIL import Image


"""
create rectangles, no images
simple transformations
"""
cur_dir='/home/projects/py/statistics/cairo_apps'
im_dir='/home/projects/Pictures'
def transform():

    height, width = 300, 300
    ar = np.zeros(shape=(height, width, 4), dtype=np.uint8)

    canvas=cairo.ImageSurface.create_for_data(ar,
                                              cairo.FORMAT_ARGB32,
                                              width,
                                              height,
                                              width * 4)

    # create a context for drawing states
    ctx=cairo.Context(canvas)

    # set color
    ctx.set_source_rgb(0.9, 0.0, 0.2)
    ctx.rectangle(0, 0, 30, 30)
    ctx.fill()

    ctx.translate(30, 30)
    ctx.set_source_rgb(0.8, 0.8, 0.2)
    ctx.rectangle(0, 0, 30, 30)
    ctx.fill()

    # net effects: 30+30, 30+30
    ctx.translate(30, 30)
    ctx.set_source_rgb(0.8, 0.1, 0.2)
    ctx.rectangle(0, 0, 30, 30)
    ctx.fill()

    im=Image.fromarray(ar, mode='RGBA')
    im.save(os.path.join(cur_dir, 'test1.jpg'))

if __name__=='__main__':
    transform()