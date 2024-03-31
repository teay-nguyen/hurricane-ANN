#!/usr/bin/env python3

from parse import load_hurricane_imgs, generate_train_set
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
from torchinfo import summary
import torch
import torch.nn as nn

if __name__ == '__main__':
  model = shufflenet_v2_x1_0(num_classes=2)
  summary(model, (1, 3, 128, 128))
  #dat = load_hurricane_imgs(shuffle_train=True)
  #X = dat['train_another']['damage']
