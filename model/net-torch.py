#!/usr/bin/env python3

print('[info] importing libraries...')
from parse import load_hurricane_imgs, generate_labels, generate_augmented_imgs,\
                  fetch_label_batch, permute_img_for_inference, permute_img_for_view
from shufnet import Net
from torchinfo import summary
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch, os

X_AUGMENTED_FILE = './saves/X_augmented.npy'
Y_AUGMENTED_FILE = './saves/Y_augmented.npy'
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

  BS, EPOCHS = 512 if torch.cuda.is_available() else 64,\
               1 if QUICK else 10
  STEPS = 50000 // BS
  model = Net(DEVICE)
  optim = torch.optim.Adam(model.parameters(), lr=1e-2)
  lossfn = nn.BCELoss()
  summary(model, (1, 3, 150, 150))

  def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
      m.eval()

  def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
      m.train()

  #    TODO: fix test accuracy
  #    is test acc flunking because of shuffling issues?
  #    batch size is not big enough

  model.cuda()
  model.apply(set_bn_train)
  losses, accuracies = [], []
  for epoch in range(EPOCHS):
    for step in range(STEPS):
      optim.zero_grad()
      batch = np.random.randint(50000, size=(BS,))
      X, Y = fetch_label_batch(batch, X_AUGMENTED_FILE, Y_AUGMENTED_FILE, DEVICE, to_tensor=True)
      Y = F.one_hot(Y.long(), num_classes=2)
      out = model(normalize(permute_img_for_inference(X)))
      cat = torch.argmax(out, dim=1)
      acc = (cat == torch.argmax(Y, dim=1)).float().mean()
      loss = lossfn(out, Y.float())
      losses.append(loss.item())
      accuracies.append(acc.item())
      loss.backward()
      optim.step()
      print(f'\repoch [{epoch+1}/{EPOCHS}] step [{step+1}/{STEPS}] loss {losses[-1]} acccuracy {accuracies[-1]}', end='')

  model.apply(set_bn_eval)
  with torch.no_grad():
    dat = load_hurricane_imgs()
    X, Y = generate_labels(dat, subset='test', shuffle=True, to_tensor=True)
    X, Y = X.to(DEVICE).float(), F.one_hot(Y.long(), num_classes=2).to(DEVICE)
    out = model(normalize(permute_img_for_inference(X)))
    cat = torch.argmax(out, dim=1)
    acc = (cat == torch.argmax(Y, dim=1)).float().mean()
    loss = lossfn(out, Y.float())
    print(f'\n[info] test accuracy {acc}')

  torch.save(model.state_dict(), MODEL_PATH)

  plt.ylim(-.1, 1.1)
  plt.plot(losses)
  plt.plot(accuracies)
  plt.show()

if __name__ == '__main__':
  train()
