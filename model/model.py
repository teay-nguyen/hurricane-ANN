#!/usr/bin/env python

from tqdm import trange
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

def load_imgs(subset: str, tensor:bool=False) -> Tuple[np.ndarray, np.ndarray] | Tuple[torch.tensor, torch.tensor]:
  dat:Optional[np.ndarray] = None; files:Optional[np.ndarray] = None; imgs:Optional[np.ndarray] = None
  assert subset in ('train_another', 'test', 'test_another', 'validation_another')
  if not os.path.isdir('./saves'): os.mkdir('./saves')
  if os.path.isfile(f'./saves/{subset}.npy'): dat = np.load(f'./saves/{subset}.npy')
  root = os.path.join("../data/images/Post-hurricane/", subset)
  process = lambda x: cv2.cvtColor(cv2.imread(x, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  if dat is None:
    files = list(map(lambda x: os.path.join(f'{root}/damage', x),    os.listdir(os.path.join(root, 'damage')))) + \
            list(map(lambda x: os.path.join(f'{root}/no_damage', x), os.listdir(os.path.join(root, 'no_damage'))))
    imgs = np.array(list(map(process, files)), dtype=np.uint8)
    np.save(f'./saves/{subset}', imgs)
  labels = ([1] * len(os.listdir(os.path.join(root, 'damage')))) + \
           ([0] * len(os.listdir(os.path.join(root, 'no_damage'))))
  labels = np.array(labels, dtype=np.uint8)
  if tensor: ret = (torch.tensor(imgs).reshape(-1, 3, 128, 128).to(torch.float), torch.tensor(labels)) if dat is None else \
                   (torch.tensor(dat).reshape(-1, 3, 128, 128).to(torch.float),  torch.tensor(labels))
  else: ret = (imgs.reshape(-1, 3, 128, 128).astype(np.float32), labels) if dat is None else \
              (dat.reshape(-1, 3, 128, 128).astype(np.float32), labels)
  return ret


# TODO: make model train

class FireModule(nn.Module):
  def __init__(self, inplanes:int, squeeze_planes:int, expand1x1_planes:int, expand3x3_planes:int):
    super(FireModule, self).__init__()
    self.inplanes = inplanes
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x:torch.tensor) -> torch.tensor:
    x = self.relu(self.squeeze(x))
    x0 = self.relu(self.expand1x1(x))
    x1 = self.relu(self.expand3x3(x))
    return torch.cat([x0, x1], 1)

class ConvNet(nn.Module):
  def __init__(self, num_classes:int=2, dropout:float=.5):
    super(ConvNet, self).__init__()
    self.num_classes = num_classes
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
      FireModule(64, 16, 64, 64),
      FireModule(128, 16, 64, 64),
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
      FireModule(128, 32, 128, 128),
      FireModule(256, 32, 128, 128),
      nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
      FireModule(256, 48, 192, 192),
      FireModule(384, 48, 192, 192),
      FireModule(384, 64, 256, 256),
      FireModule(512, 64, 256, 256),
    )

    self.classifier = nn.Sequential(
      nn.Dropout(p=dropout),
      nn.Conv2d(512, num_classes, kernel_size=1),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d((1,1))
    )

  def forward(self, x:torch.tensor) -> torch.tensor:
    x = self.features(x)
    x = self.classifier(x)
    return torch.flatten(x, 1)

def shuffle(x:Union[np.ndarray, torch.tensor], y:Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
  perm = np.random.permutation(x.shape[0])
  return x[perm], y[perm]

if __name__ == '__main__':
  BS = 32
  lrs = [1e-3, 1e-4, 1e-5, 1e-5]
  epochss = [13, 3, 3, 1]
  model = ConvNet(num_classes=2)
  X_train, Y_train = load_imgs('train_another', tensor=True)
  X_train, Y_train = shuffle(X_train, Y_train)
  steps = X_train.shape[0] // BS
  print(type(X_train), X_train.shape)

  samps = np.random.randint(0, X_train.shape[0], size=(BS))
  X, Y = X_train[samps], Y_train[samps]

  out = model(X)
  print(out.shape)
