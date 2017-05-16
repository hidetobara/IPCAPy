
import configure
import define

import glob
import os
from PIL import Image, ImageFilter

def run_edge():
    index = 0
    for path in glob.glob(define.get_data_path() + "porn/*.jpg"):
        #dir = os.path.dirname(path)
        #name, extension = os.path.splitext(os.path.basename(path))

        img = Image.open(path)
        minlen = min(img.size[0], img.size[1])
        if minlen < 128: continue
        if img.mode != "RGB": img = img.convert("RGB")

        orgpath = define.get_data_path() + "tmp/" + str(index) + "-org.png"
        newpath = define.get_data_path() + "tmp/" + str(index) + "-abs.png"
        index += 1

        img.save(orgpath)
        img = img.filter(ImageFilter.FIND_EDGES)
        img = img.convert("L")
        img = img.point(lambda x: int(x / 32) * 32)
        img.save(newpath)

def trim_square(img, size):
    size += 0.1
    minlen = min(img.size[0], img.size[1])
    newsize = (int(img.size[0] * size / minlen), int(img.size[1] * size / minlen))
    img = img.resize(newsize, Image.BILINEAR)
    left, upper = int((newsize[0] - size) / 2), int((newsize[1] - size) / 2)
    img = img.crop((left, upper, left + size, upper + size))
    return img

run_edge()