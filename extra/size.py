#!/usr/bin/env python3

import os

if __name__ == '__main__':
  file_cnt = 0
  for d in os.listdir('../data/HURSAT'):
    dp = os.path.join('../data/HURSAT', d)
    file_cnt += len(os.listdir(dp))
  print(f'file count: {file_cnt}')
