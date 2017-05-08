#!/usr/bin/env python

from __future__ import print_function

import define
import os
import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

class FacadeUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        super(FacadeUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, enc)
        return loss
        
    def loss_dec(self, dec, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, dec)
        return loss
        
        
    def loss_dis(self, dis, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape
        
        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):        
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')
        
        enc, dec, dis = self.enc, self.dec, self.dis
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]
        w_in = 128
        w_out = 128
        #print(batch[0][0].shape, batch[0][1].shape)
        
        x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")
        
        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)

        #print(x_in.shape)
        z = enc(x_in, test=False)
        x_out = dec(z, test=False)

        y_fake = dis(x_in, x_out, test=False)
        y_real = dis(x_in, t_out, test=False)


        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake)
        for z_ in z:
            z_.unchain_backward()
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake)
        x_in.unchain_backward()
        x_out.unchain_backward()
        dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)

    def generate(self, paths, dir):
        enc, dec = self.enc, self.dec
        xp = enc.xp
        in_ch = define.get_in_ch()
        w_in = 128
        w_out = 128

        x_in = xp.zeros((len(paths), in_ch, w_in, w_in)).astype("f")
        for i,path in enumerate(paths):
            print("load=" + path)
            label = Image.open(path)
            w, h = label.size
            r = 128 / min(w, h)
            label = label.resize((int(r * w), int(r * h)), Image.NEAREST)

            label_ = np.asarray(label) / (256/in_ch)
            label = np.zeros((in_ch, label.size[0], label.size[1])).astype("i")
            for j in range(in_ch):
                label[j, :] = label_ == j
            x_in[i,:] = xp.asarray(label)

        x_in = Variable(x_in)
        z = enc(x_in, test=False)
        x_out = dec(z, test=False)

        for i,path in enumerate(paths):
            o = chainer.cuda.to_cpu(x_out[i].data)
            o = np.asarray(np.clip(o * 128 + 128, 0.0, 255.0), dtype=np.uint8).transpose((1,2,0)).reshape((w_out,w_out,3))
            filename = os.path.basename(path)
            genpath = dir + "/" + filename
            Image.fromarray(o).convert('RGB').save(genpath)




