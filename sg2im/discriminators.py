#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sg2im.bilinear import crop_bbox_batch
from sg2im.layers import GlobalAvgPool, Flatten, get_activation, build_cnn


class PatchDiscriminator(nn.Module):
  def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
               padding='same', pooling='avg', input_size=(128,128),
               layout_dim=0):
    super(PatchDiscriminator, self).__init__()
    input_dim = 3 + layout_dim
    arch = 'I%d,%s' % (input_dim, arch)
    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.cnn, output_dim = build_cnn(**cnn_kwargs)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

  def forward(self, x, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    return self.cnn(x)


class AcDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='none', activation='relu',
               padding='same', pooling='avg'):
    super(AcDiscriminator, self).__init__()
    self.vocab = vocab

    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling, 
      'padding': padding,
    }
    cnn, D = build_cnn(**cnn_kwargs)
    self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
    num_objects = len(vocab['object_idx_to_name'])

    self.real_classifier = nn.Linear(1024, 1)
    self.obj_classifier = nn.Linear(1024, num_objects)

  def forward(self, x, y):
    if x.dim() == 3:
      x = x[:, None]
    vecs = self.cnn(x)
    real_scores = self.real_classifier(vecs)
    obj_scores = self.obj_classifier(vecs)
    ac_loss = F.cross_entropy(obj_scores, y)
    return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='none', activation='relu',
               object_size=64, padding='same', pooling='avg'):
    super(AcCropDiscriminator, self).__init__()
    self.vocab = vocab
    self.discriminator = AcDiscriminator(vocab, arch, normalization,
                                         activation, padding, pooling)
    self.object_size = object_size

  def forward(self, imgs, objs, boxes, obj_to_img):
    crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
    real_scores, ac_loss = self.discriminator(crops, objs)
    return real_scores, ac_loss


# Multi-scale discriminator as in pix2pixHD or SPADE
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            for j in range(n_layers + 2):
                setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        result = [input]
        for i in range(len(model)):
            result.append(model[i](result[-1]))
        return result[1:]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                     range(self.n_layers + 2)]
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]


def divide_pred(pred):
  # the prediction contains the intermediate outputs of multiscale GAN,
  # so it's usually a list
  if type(pred) == list:
      fake = []
      real = []
      for p in pred:
          fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
          real.append([tensor[tensor.size(0) // 2:] for tensor in p])
  else:
      fake = pred[:pred.size(0) // 2]
      real = pred[pred.size(0) // 2:]

  return fake, real