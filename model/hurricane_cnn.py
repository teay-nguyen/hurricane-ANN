#!/usr/bin/env python

import torchvision.transforms as transforms
from typing import Tuple
from tqdm import trange
from tinygrad import Tensor
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

# TODO: refactor into one func

def load_train_data() -> Tuple[Tensor, Tensor]:
  if os.path.isfile('./saves/train_another.npz'):
    dat = np.load('./saves/train_another.npz')
    return Tensor(dat['X_train']), Tensor(dat['Y_train'])

  dmg, no_dmg = load_imgs('train_another')

  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomVerticalFlip(p=.5),
    transforms.RandomRotation(90),
    transforms.RandomCrop(size=(128, 128)),
    transforms.ToTensor()
  ])

  dmg_transformed_imgs = np.array([transform(dmg_img).cpu().detach().numpy().swapaxes(0,2) for dmg_img in dmg], dtype=np.float32)
  nodmg_transformed_imgs = np.array([transform(nodmg_img).cpu().detach().numpy().swapaxes(0,2) for nodmg_img in no_dmg], dtype=np.float32)

  dmg_train = np.concatenate((dmg, dmg_transformed_imgs), axis=0)
  nodmg_train = np.concatenate((no_dmg, nodmg_transformed_imgs), axis=0)

  X_train = np.concatenate((dmg_train, nodmg_train), axis=0)
  Y_train = np.array([1 for _ in range(dmg_train.shape[0])]+[0 for _ in range(nodmg_train.shape[0])], dtype=np.uint8)
  np.savez('./saves/train_another.npz', X_train=X_train, Y_train=Y_train)

  return Tensor(X_train), Tensor(Y_train)

def load_val_data() -> Tuple[Tensor, Tensor]:
  if os.path.isfile('./saves/validation_another.npz'):
    dat = np.load('./saves/validation_another.npz')
    return Tensor(dat['X_val']), Tensor(dat['Y_val'])

  dmg, no_dmg = load_imgs('validation_another')

  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomVerticalFlip(p=.5),
    transforms.RandomRotation(90),
    transforms.RandomCrop(size=(128, 128)),
    transforms.ToTensor()
  ])

  dmg_transformed_imgs = np.array([transform(dmg_img).cpu().detach().numpy().swapaxes(0,2) for dmg_img in dmg], dtype=np.float32)
  nodmg_transformed_imgs = np.array([transform(nodmg_img).cpu().detach().numpy().swapaxes(0,2) for nodmg_img in no_dmg], dtype=np.float32)

  dmg_val = np.concatenate((dmg, dmg_transformed_imgs), axis=0)
  nodmg_val = np.concatenate((no_dmg, nodmg_transformed_imgs), axis=0)

  X = np.concatenate((dmg_val, nodmg_val), axis=0)
  Y = np.array([1 for _ in range(dmg_val.shape[0])]+[0 for _ in range(nodmg_val.shape[0])], dtype=np.uint8)
  np.savez('./saves/validation_another.npz', X_val=X, Y_val=Y)

  return Tensor(X), Tensor(Y)

def load_test_data(subset: str) -> Tuple[Tensor, Tensor]:
  assert subset in ('test', 'test_another')
  if os.path.isfile(f'./saves/{subset}.npz'):
    dat = np.load(f'./saves/{subset}.npz')
    return Tensor(dat['X_test']), Tensor(dat['Y_test'])

  dmg, no_dmg = load_imgs(subset)

  X_test = np.concatenate((dmg, no_dmg), axis=0)
  Y_test = np.array([1 for _ in range(dmg.shape[0])]+[0 for _ in range(no_dmg.shape[0])], dtype=np.uint8)
  np.savez(f'./saves/{subset}.npz', X_test=X_test, Y_test=Y_test)

  return Tensor(X_test), Tensor(Y_test)

if __name__ == '__main__':
  # X_train, Y_train = load_train_data()
  pass