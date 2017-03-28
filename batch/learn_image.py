import configure
import define
from PIL import Image
import numpy
import glob
import IncrementalPCA
import IncrementalPCAConvolution


def run_ipca(ite=15):
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
    cipca = IncrementalPCAConvolution.IncrementalPCAConvolution(16, (32,32,3))
    cipca.load(define.get_data_path() + "cat.cipca")

    for path in glob.glob(define.get_data_path() + "cat/*.jpg"):
        print("\t" + path)
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((64, 64), Image.ANTIALIAS)
        array = numpy.asarray(img, dtype=numpy.float64) / 255.0
        cipca.fit(array)
    cipca.save(define.get_data_path() + "cat.cipca")
    print(cipca._ipca.evaluate_basis())

run_cipca()