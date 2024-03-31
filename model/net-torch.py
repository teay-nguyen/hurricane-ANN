#!/usr/bin/env python3

from parse import load_hurricane_imgs, generate_labels
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
from torchinfo import summary
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

TRANSFORM = transforms.Compose([
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )
])

def save_model(epochs:int, model, optim, lossfn) -> None:
  torch.save({
    'epochs':epochs,
    'model_sd':model.state_dict(),
    'optim_sd':optim.state_dict(),
    'loss':lossfn
  }, './saves/model.pth')

if __name__ == '__main__':
  LR = 0.003
  model = shufflenet_v2_x1_0(num_classes=2)
  optim = torch.optim.Adam(model.parameters(), lr=LR)
  lossfn = nn.CrossEntropyLoss()
  summary(model, (1, 3, 128, 128))
  dat = load_hurricane_imgs()
  X_train, Y_train = generate_labels(dat, subset='train_another', shuffle=True, to_tensor=True)
  X_val, Y_val = generate_labels(dat, subset='validation_another', shuffle=True, to_tensor=True)
  BS, EPOCHS = 512, 20
  STEPS = X_train.shape[0]//BS

  losses, accuracies = [], []
  for epoch in range(EPOCHS):
    for step in range(STEPS):
      model.train()
      samps = torch.randint(0, X_train.shape[0], (BS,))
      X, Y = TRANSFORM(X_train[samps]), Y_train[samps]
      optim.zero_grad()
      out = model(X)
      loss = lossfn(out, Y)
      cat = torch.argmax(out, dim=1)
      acc = (cat.numpy() == Y.numpy()).mean()
      loss.backward()
      optim.step()

      model.eval()
      with torch.no_grad():
        X, Y = TRANSFORM(X_val), Y_val
        out = model(X)
        val_loss = lossfn(out, Y)
        cat = torch.argmax(out, dim=1)
        val_acc = (cat.numpy() == Y.numpy()).mean()

      print(f'\repoch [{epoch+1}/{EPOCHS}] step [{step+1}/{STEPS}] loss {loss.item():.6f} accuracy {acc*100:.3f}% val_loss {val_loss.item():.6f} val_accuracy {val_acc*100:.3f}%', end='')

      losses.append(loss.item())
      accuracies.append(acc)
    print()
  save_model(EPOCHS, model, optim, lossfn)
  plt.ylim(-0.1, 1.1)
  plt.plot(losses)
  plt.plot(accuracies)
  plt.show()