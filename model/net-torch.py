#!/usr/bin/env python

from parse import load_hurricane_imgs, generate_labels, HurricaneImages
from torchvision.transforms import v2
from torchinfo import summary
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.grid'] = False

class Net(torch.nn.Module):
  def __init__(self):
    pass

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return x

def run_train() -> None:
  pass

# TODO: transform doesn't work on 4d batches

if __name__ == '__main__':
  x, y = generate_labels(load_hurricane_imgs(), subset='train_another', to_tensor=True)
  imgs = HurricaneImages(x, y, transform=v2.Resize((150, 150)))
  xi, yi = imgs[0]
  plt.imshow(xi.permute(1, 2, 0))
  plt.show()
  #x = upscale_data(x.permute(0, 3, 1, 2))