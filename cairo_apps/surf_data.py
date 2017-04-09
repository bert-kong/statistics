
"""
get data from cairo surface
and import data into numpy array
"""

import numpy as np
from PIL import Image
import cairo
import os

im_dir='/home/projects/Pictures'

def test():
    import StringIO

    im=Image.open(os.path.join(im_dir, "man_face.jpeg"))
    buf=StringIO.StringIO()
    im.save(buf, format='png')
    buf.seek(0)

    surf=cairo.ImageSurface.create_from_png(buf)

    w=surf.get_width()
    h=surf.get_height()
    stride=surf.get_stride()
    string_buf=surf.get_data()

    ar=np.fromstring(string_buf, dtype=np.uint8).reshape(h, w, 4)

    im=Image.fromarray(ar, mode="RGBA")
    im.save("test.png")


if __name__=='__main__':
    test()



