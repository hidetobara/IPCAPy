
import configure
import define

import glob
import os
from PIL import Image, ImageFilter

def run_edge():
    for path in glob.glob(define.get_data_path() + "cat/*.jpg"):
        img = Image.open(path)
        img.thumbnail((128, 128), Image.ANTIALIAS)
        img = img.filter(ImageFilter.FIND_EDGES)
        img = img.convert("L")
        img = img.point(lambda x: int(x / 32) * 32)

        dir = os.path.dirname(path)
        name, extension = os.path.splitext(os.path.basename(path))
        newpath = define.get_data_path() + name + "-edge.png"
        img.save(newpath)

run_edge()