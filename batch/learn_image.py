import configure
import define
from PIL import Image
import numpy
import glob
import os
import IncrementalPCA
import IncrementalPCAConvolution


def run_ipca(ite=5):
    ipca = IncrementalPCA.IncrementalPCA(16, 32*32*3)
    ipca.load(define.get_data_path() + "cat.ipca")

    for n in range(ite):
        for path in glob.glob(define.get_data_path() + "cat/*.jpg"):
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((32, 32), Image.ANTIALIAS)
            width, height = img.size
            array = numpy.asarray(img, dtype=numpy.float64) / 255.0
            ipca.fit(array)
        # 外れ値の計算
        print("outlier: " + str(ipca.evaluate_outlier(array, is_refreshing=True)))
    # モデルの保存
    ipca.save(define.get_data_path() + "cat.ipca", (32,32,3))

    # データの圧縮復元
    compress = ipca.transform(array)
    restored = ipca.inv_transform(compress) * 255.0
    img = Image.fromarray(numpy.uint8(restored.reshape(32,32,3)))
    img.save(define.get_data_path() + "reflect.png")

    # 基底の評価
    evaluated = ipca.evaluate_basis()
    print(evaluated)

def run_cipca():
    cipca = IncrementalPCAConvolution.IncrementalPCAConvolution(32, (16,16,3))
    cipca.load(define.get_data_path() + "cat.cipca")

    for path in glob.glob(define.get_data_path() + "cat/*.jpg"):
        print("\t" + path)
        filename = os.path.basename(path)
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((64, 64), Image.ANTIALIAS)
        array = numpy.asarray(img, dtype=numpy.float64) / 255.0
        cipca.fit(array)
        continue

        # 強い成分を保存
        compress = cipca.transform(array)
        shape = compress.shape
        #print(shape, compress[1,1])
        tmp = numpy.argmax(compress.reshape((shape[0]*shape[1],shape[2])), axis=1)
        tmp = numpy.uint8(tmp.reshape(shape[0], shape[1])*8)
        #print(tmp.shape, tmp[1,1])
        img = Image.fromarray(tmp, mode="L")
        img.save(define.get_data_path() + "cmp." + filename + ".png")
    cipca.save(define.get_data_path() + "cat.cipca")
    print(cipca._ipca.evaluate_basis())

run_cipca()
