#!/usr/bin/env python

from typing import List, Callable, Tuple
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import cv2, os

plt.rcParams['savefig.bbox'] = 'tight'
np.random.seed(1337)

def load_imgs(label: str) -> Tuple[np.ndarray, np.ndarray]:
  dmg, no_dmg = [], []
  ROOT = '../data/images/Post-hurricane'
  LABEL_PATH = f'{ROOT}/{label}'
  DMG_PATH = f'{LABEL_PATH}/damage'
  NO_DMG_PATH = f'{LABEL_PATH}/no_damage'

  name_arr = os.listdir(DMG_PATH)
  for i in (t := trange(len(name_arr))):
    t.set_description(name_arr[i])
    img_path = os.path.join(DMG_PATH, name_arr[i])
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    dmg.append((img[:,:,:] / 255.).astype(np.float32))

  name_arr = os.listdir(NO_DMG_PATH)
  for i in (t := trange(len(name_arr))):
    t.set_description(name_arr[i])
    img_path = os.path.join(NO_DMG_PATH, name_arr[i])
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    no_dmg.append((img[:,:,:] / 255.).astype(np.float32))

  return np.array(dmg, dtype=np.float32), np.array(no_dmg, dtype=np.float32)

if __name__ == '__main__':
  dmg, no_dmg = load_imgs('test')
  print(dmg.shape, no_dmg.shape)