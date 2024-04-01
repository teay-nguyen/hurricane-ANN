#!/usr/bin/env python3

from parse import load_hurricane_imgs, generate_labels
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
from torchinfo import summary
from torchvision import transforms
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

if not 'QUICK' in os.environ:
  os.environ['QUICK'] = '0'
QUICK = True if os.environ['QUICK'] == '1' else False
print('QUICK', os.environ['QUICK'])

AUGMENT = transforms.Compose([
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

def save_model(epochs:int, model, optim, lossfn) -> None:
  torch.save({
    'epochs':epochs,
    'model_sd':model.state_dict(),
    'optim_sd':optim.state_dict(),
    'loss':lossfn
  }, './saves/model.pth')

def run_train() -> None:
  LR = 0.004
  model = shufflenet_v2_x1_0(num_classes=2)
  optim = torch.optim.Adam(model.parameters(), lr=LR)
  lossfn = nn.CrossEntropyLoss()
  summary(model, (1, 3, 128, 128))
  dat = load_hurricane_imgs()
  X_train, Y_train = generate_labels(dat, subset='train_another', shuffle=True, to_tensor=True)
  X_test, Y_test = generate_labels(dat, subset='test', shuffle=True, to_tensor=True)
  BS, EPOCHS = 256, (20 if not QUICK else 3)
  STEPS = X_train.shape[0] // BS

  losses, accuracies = [], []
  for epoch in range(EPOCHS):
    for step in range(STEPS):
      samps = torch.randint(0, X_train.shape[0], (BS,))
      X_aug, Y = AUGMENT(X_train[samps]), Y_train[samps]
      optim.zero_grad()
      out = model(X_aug)
      loss = lossfn(out, Y)
      cat = torch.argmax(out, dim=1)
      acc = (cat.numpy() == Y.numpy()).astype(np.float32).mean()
      loss.backward()
      optim.step()
      losses.append(loss.item())
      accuracies.append(acc)
      print(f'\repoch [{epoch+1}/{EPOCHS}] step [{step+1}/{STEPS}] loss {loss.item():.6f} accuracy {acc*100:.3f}%', end='')
    print()

  with torch.no_grad():
    out = model(X_test)
  cat = torch.argmax(out, dim=1)
  acc = (cat.numpy() == Y_test.numpy()).astype(np.float32).mean()
  print(f'test accracy {acc*100}%')

  save_model(EPOCHS, model, optim, lossfn)
  plt.ylim(-0.1, 1.1)
  plt.plot(losses)
  plt.plot(accuracies)
  plt.show()

if __name__ == '__main__':
  run_train()
