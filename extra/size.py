#!/usr/bin/env python3

import os
import sys

def count_files(dp: str):
  cnt = 0
  for ch in os.listdir(dp):
    pth = os.path.join(dp, ch)
    if os.path.isfile(pth): cnt += 1
    elif os.path.isdir(pth): cnt += count_files(pth)
  return cnt

if __name__ == '__main__':
  print('file count:', count_files('../data/images'))
