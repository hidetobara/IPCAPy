import configure
import define
from PIL import Image
import numpy
import glob
import IncrementalPCA


def run():
    ipca = IncrementalPCA.IncrementalPCA(32*32*3, 16)
    ipca.load(define.get_data_path() + "cat.ipca")

    for path in glob.glob(define.get_data_path() + "cat/*.jpg"):
        print(path)
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((32, 32), Image.ANTIALIAS)
        width, height = img.size
        array = numpy.asarray(img, dtype=numpy.float64)
        #print(array.shape, array.dtype)
        row = []
        for h in range(height):
            for w in range(width):
                row.append(array[h,w][0])
                row.append(array[h,w][1])
                row.append(array[h,w][2])
        ipca.fit(numpy.asarray(row, dtype=numpy.float64))
    ipca.save(define.get_data_path() + "cat.ipca")


run()