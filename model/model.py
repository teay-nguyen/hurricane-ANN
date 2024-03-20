#!/usr/bin/env python

from typing import Tuple, List, Callable
from tinygrad import Tensor, TinyJit, nn
from tqdm import trange

import cv2, os
import numpy as np
import matplotlib.pyplot as plt

def load_imgs(subset: str, tensor=False) -> Tuple[np.ndarray, np.ndarray]:
  dat, files, imgs = None, None, None
  assert subset in ('train_another', 'test', 'test_another', 'validation_another')
  if not os.path.isdir('./saves'): os.mkdir('./saves')
  if os.path.isfile(f'./saves/{subset}.npy'): dat = np.load(f'./saves/{subset}.npy')
  root = os.path.join("../data/images/Post-hurricane/", subset)
  process = lambda x: cv2.cvtColor(cv2.imread(x, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  if dat is None:
    files = list(map(lambda x: os.path.join(f'{root}/damage', x),    os.listdir(os.path.join(root, 'damage')))) + \
            list(map(lambda x: os.path.join(f'{root}/no_damage', x), os.listdir(os.path.join(root, 'no_damage'))))
    imgs = np.array(list(map(process, files)), dtype=np.uint16)
    np.save(f'./saves/{subset}', imgs)
  labels = ([1] * len(os.listdir(os.path.join(root, 'damage')))) + \
           ([0] * len(os.listdir(os.path.join(root, 'no_damage'))))
  labels = np.array(labels, dtype=np.uint8)
  if tensor: ret = (Tensor(imgs).reshape(-1, 3, 128, 128), Tensor(labels)) if dat is None else (Tensor(dat).reshape(-1, 3, 128, 128), Tensor(labels))
  else: ret = (imgs, labels) if dat is None else (dat, labels)
  return ret

# TODO: make model train

class ConvNet:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(3, 96, 11, stride=4), Tensor.relu,
      Tensor.max_pool2d,
      nn.Conv2d(96, 256, 5, stride=1, padding=2), Tensor.relu,
      Tensor.max_pool2d,
      nn.Conv2d(256, 384, 3, stride=1, padding=1), Tensor.relu,
      nn.Conv2d(384, 384, 3, stride=1, padding=1), Tensor.relu,
      nn.Conv2d(384, 256, 3, stride=1, padding=1), Tensor.relu,
      Tensor.max_pool2d,
      Tensor.dropout,
      lambda x: x.flatten(1),
      nn.Linear(2304, 4096), Tensor.relu,
      Tensor.dropout,
      nn.Linear(4096, 4096), Tensor.relu,
      nn.Linear(4096, 2), Tensor.softmax,
    ]

  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential(self.layers)

if __name__ == '__main__':
  imgs, labels = load_imgs('train_another', tensor=True)
  model = ConvNet()

  out = (model(imgs).argmax(axis=1) == labels).mean()*100
