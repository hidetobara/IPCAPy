import os

import define
import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np

import skimage.io as io

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./facade/base', data_range=(1,300)):
        print("load dataset start")
        print("    from: %s"%dataDir)
        print("    range: [%d, %d)"%(data_range[0], data_range[1]))
        self.dataDir = dataDir
        self.dataset = []
        for i in range(data_range[0],data_range[1]):
            img = Image.open(dataDir+"/cmp_b%04d.jpg"%i)
            label = Image.open(dataDir+"/cmp_b%04d.png"%i)
            w,h = img.size
            r = 286/min(w,h)
            # resize images so that min(w, h) == 286
            img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            label = label.resize((int(r*w), int(r*h)), Image.NEAREST)
            
            img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
            label_ = np.asarray(label)-1  # [0, 12)
            label = np.zeros((12, img.shape[1], img.shape[2])).astype("i")
            for j in range(12):
                label[j,:] = label_==j
            self.dataset.append((img,label))
        print("load dataset done", img.shape, label.shape)
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i):
        crop_width = define.get_nn_size()
        _,h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width
        return self.dataset[i][1][:,y_l:y_r,x_l:x_r], self.dataset[i][0][:,y_l:y_r,x_l:x_r]
    
class DecompDataset(FacadeDataset):
    def __init__(self, dataDir='./', data_range=(1, 300), shift=1000):
        print("load dataset start")
        self.dataDir = dataDir
        self.dataset = []
        for i in range(data_range[0], data_range[1]):
            img = Image.open(dataDir + "/%d-org.png" % i)
            lbl = Image.open(dataDir + "/%d-abs.png" % i)
            size = min(img.size.width, img.size.height)
            while size > 128:
                self.append(img, lbl, size)
                size -= shift
                shift = shift * 2
        print("load dataset done", len(self.dataset), self.dataset[0][0].shape)

    def append(self, image, label, size):
        in_ch = define.get_in_ch()
        w, h = image.size
        r = size / min(w, h)
        i = image.resize((int(r * w), int(r * h)), Image.BILINEAR)
        l = label.resize((int(r * w), int(r * h)), Image.NEAREST)

        i = np.asarray(i).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        l = np.asarray(l) / int(256 / in_ch)
        lc = np.zeros((in_ch, i.shape[1], i.shape[2])).astype("i")
        for j in range(in_ch):
            lc[j, :] = l == j
        self.dataset.append((i, lc))
