
import configure
import define

import glob
import os
from PIL import Image, ImageFilter

def run_edge():
    for index,path in enumerate(glob.glob(define.get_data_path() + "test/*.jpg")):
        #dir = os.path.dirname(path)
        #name, extension = os.path.splitext(os.path.basename(path))
        orgpath = define.get_data_path() + str(index) + "-org.png"
        newpath = define.get_data_path() + str(index) + "-abs.png"

        img = Image.open(path)
        img.thumbnail((128, 128), Image.ANTIALIAS)
        img.save(orgpath)
        img = img.filter(ImageFilter.FIND_EDGES)
        img = img.convert("L")
        img = img.point(lambda x: int(x / 32) * 32)
        img.save(newpath)

run_edge()