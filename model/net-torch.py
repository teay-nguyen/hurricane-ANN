#!/usr/bin/env python3

print('[info] importing libraries...')
from parse import load_hurricane_imgs, generate_labels, generate_augmented_imgs,\
                  fetch_label_batch, permute_img_for_train, permute_img_for_view
from shufnet import Net

from torchinfo import summary
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch, os

X_FILE = './saves/X_augmented.npy'
Y_FILE = './saves/Y_augmented.npy'
MODEL_PATH = "./saves/model_saves/model.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'[info] device {DEVICE} type {type(DEVICE)}')
print(f'[info] device count {torch.cuda.device_count()}')

def train():
  if not 'QUICK' in os.environ:
    os.environ['QUICK'] = '0'
  QUICK = True if os.environ['QUICK'] == '1' else False

  normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

  BS, EPOCHS = 256 if torch.cuda.is_available() else 64,\
               1 if QUICK else 10
  STEPS = 50000 // BS
  model = Net(DEVICE)
  optim = torch.optim.Adam(model.parameters(), lr=0.04)
  lossfn = nn.CrossEntropyLoss()
  summary(model, (1, 3, 150, 150))

  model.cuda()
  model.train()
  losses, accuracies = [], []
  for epoch in range(EPOCHS):
    for step in range(STEPS):
      optim.zero_grad()
      batch = np.random.randint(50000, size=(BS,))
      X, Y = fetch_label_batch(batch, X_FILE, Y_FILE, DEVICE, to_tensor=True)
      out = model(normalize(permute_img_for_train(X)))
      cat = torch.argmax(out, dim=1)
      acc = (cat.cpu().numpy() == Y.cpu().numpy()).astype(np.float32).mean()
      loss = lossfn(out, Y)
      losses.append(loss)
      accuracies.append(acc)
      loss.backward()
      optim.step()

      print(f'\repoch [{epoch+1}/{EPOCHS}] step [{step+1}/{STEPS}] loss {losses[-1].item()} acccuracy {accuracies[-1]}', end='')

  torch.save(model.state_dict(), MODEL_PATH)

if __name__ == '__main__':
  train()
