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

def load_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    size = define.get_pca_size()
    img.thumbnail((size,size), Image.ANTIALIAS)
    array = numpy.asarray(img, dtype=numpy.float64) / 255.0
    return array

def save_image(path, image):
    size = define.get_pca_size()
    img = Image.fromarray(numpy.uint8(image.reshape(size,size,3) * 255.0))
    img.save(path)

def run_cipca1():
    cipca1 = IncrementalPCAConvolution.IncrementalPCAConvolution(24, (16,16,3))
    cipca1.load(define.get_data_path() + "cat.cipca1")

    images = []
    print("1-1.")
    for n,path in enumerate(glob.glob(define.get_data_path() + "test/*.jpg")):
        image = load_image(path)
        save_image(define.get_data_path() + str(n) + "-org.png", image)
        images.append(image)
    print("1-2.")
    for image in images:
        cipca1.fit(image)
    print("1-3.")
    for n,image in enumerate(images):
        t = cipca1.transform(image)
        shape = t.shape
        #print(shape, t[1,1])
        tmp = numpy.argmax(t.reshape((shape[0]*shape[1],shape[2])), axis=1)
        tmp = numpy.uint8(tmp.reshape(shape[0], shape[1])*10)
        #print(tmp.shape, tmp[1,1])
        img = Image.fromarray(tmp, mode="L")
        img.save(define.get_data_path() + str(n) + "-abs.png")
    cipca1.save(define.get_data_path() + "cat.cipca1")

def run_cipca3():
    cipca1 = IncrementalPCAConvolution.IncrementalPCAConvolution(16, (16,16,3))
    cipca1.load(define.get_data_path() + "cat.cipca1")
    cipca2 = IncrementalPCAConvolution.IncrementalPCAConvolution(32, (16,16,16))
    cipca2.load(define.get_data_path() + "cat.cipca2")
    cipca3 = IncrementalPCAConvolution.IncrementalPCAConvolution(64, (8,8,32))
    cipca3.load(define.get_data_path() + "cat.cipca3")

    images = []
    print("3-1.")
    for n,path in enumerate(glob.glob(define.get_data_path() + "cat/*.jpg")):
        image = load_image(path)
        save_image(define.get_data_path() + "org-" + str(n) + ".png", image)
        images.append(image)

    print("3-2.")
    for image in images:
        cipca1.fit(image)
    cipca1.save(define.get_data_path() + "cat.cipca1")
    print("3-3.")
    transformed = []
    for image in images:
        transformed.append(cipca1.transform(image))
    images = transformed

    print("3-4.")
    for image in images:
        cipca2.fit(image)
    cipca2.save(define.get_data_path() + "cat.cipca2")
    print("3-5.")
    transformed = []
    for image in images:
        transformed.append(cipca2.transform(image))
    images = transformed

    print("3-6.")
    for image in images:
        cipca3.fit(image)
    print("3-7.")
    for n,image in enumerate(images):
        t = cipca3.transform(image)
        shape = t.shape
        #print(shape, t[1,1])
        tmp = numpy.argmax(t.reshape((shape[0]*shape[1],shape[2])), axis=1)
        tmp = numpy.uint8(tmp.reshape(shape[0], shape[1])*4)
        #print(tmp.shape, tmp[1,1])
        img = Image.fromarray(tmp, mode="L")
        img.save(define.get_data_path() + "gen-" + str(n) + ".png")
    cipca3.save(define.get_data_path() + "cat.cipca3")

run_cipca1()
