#!/usr/bin/env python

from typing import Tuple

import cv2, os
import numpy as np
import matplotlib.pyplot as plt

def load_imgs(subset: str) -> Tuple[np.ndarray, np.ndarray]:
  assert subset in ('train_another', 'test', 'test_another', 'validation_another')
  if os.path.isfile(f'./saves/{subset}.npz'):
    dat = np.load(f'./saves/{subset}.npz')
    return dat['imgs'], dat['labels']
  root = os.path.join("../data/images/Post-hurricane/", subset)
  process = lambda x: cv2.cvtColor(cv2.imread(x, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  files = list(map(lambda x: os.path.join(f'{root}/damage', x),    os.listdir(os.path.join(root, 'damage')))) + \
          list(map(lambda x: os.path.join(f'{root}/no_damage', x), os.listdir(os.path.join(root, 'no_damage'))))
  labels = ([1] * len(os.listdir(os.path.join(root, 'damage')))) + \
           ([0] * len(os.listdir(os.path.join(root, 'no_damage'))))
  imgs = np.array(list(map(process, files)), dtype=np.uint16)
  labels = np.array(labels, dtype=np.uint8)
  np.savez(f'./saves/{subset}.npz', imgs=imgs, labels=labels)
  return imgs, labels

# TODO: make model train

if __name__ == '__main__':
  for subset in ('test', 'test_another', 'train_another', 'validation_another'):
    x, y = load_imgs(subset)
    print(x.shape, y.shape)
