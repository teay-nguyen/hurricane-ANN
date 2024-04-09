#!/usr/bin/env python3

print('[info] importing libraries...')
from parse import load_hurricane_imgs, generate_labels, generate_augmented_imgs,\
                  fetch_label_batch, permute_img_for_train, permute_img_for_view
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

X_FILE = './saves/X_augmented.npy'
Y_FILE = './saves/Y_augmented.npy'

class SimpleNet(nn.Module):
  def __init__(self):
    super(SimpleNet, self).__init__()
    self.conv_layers = nn.Sequential(
      nn.Conv2d(3, 16, 3), nn.MaxPool2d(2, stride=2),
      nn.Conv2d(16, 3, 3), nn.MaxPool2d(2, stride=2),
      nn.Conv2d(3, 3, 3), nn.MaxPool2d(2, stride=2))

    self.fc1 = nn.Linear(867, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    x = self.conv_layers(x)
    x = x.flatten(1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return x

if __name__ == '__main__':
  model = SimpleNet()
  summary(model)