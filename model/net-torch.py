#!/usr/bin/env python3

print('[info] importing libraries...')
from parse import load_hurricane_imgs, generate_labels, generate_augmented_imgs

if __name__ == '__main__':
  dat = load_hurricane_imgs()
  X_train, Y_train = generate_labels(dat, subset='train_another', shuffle=True, to_tensor=False)
  generate_augmented_imgs(X_train, Y_train)
