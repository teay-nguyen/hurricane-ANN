#!/usr/bin/env python3

print('[info] importing libraries...')
from parse import load_hurricane_imgs, generate_labels, generate_augmented_imgs
from torchvision.transforms import v2
import matplotlib.pyplot as plt

if __name__ == '__main__':
  # load hurricane dataset
  print('[info] loading dataset...')
  dat = load_hurricane_imgs()
  X_train, Y_train = generate_labels(dat, subset='train_another', shuffle=True, to_tensor=True)
  X_val, Y_val = generate_labels(dat, subset='validation_another', shuffle=False, to_tensor=True)

  # upscale, permute and generate augmented images
  X_train, Y_train = generate_augmented_imgs(X_train, Y_train, scale=True, shuffle=True)
  plt.imshow(X_train[0])
  plt.show()