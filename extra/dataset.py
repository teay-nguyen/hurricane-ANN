#!/usr/bin/env python3

import numpy as np

def download_HURSAT_zip(url: str):
  root = "../data/HURSAT/"
  from bs4 import BeautifulSoup
  import os, requests, tarfile
  from tqdm import trange

  rep = requests.get(url)
  soup = BeautifulSoup(rep.content, 'html.parser')

  _links, _hrefs, links, hrefs = [], [], [], []
  for anchor in soup.find_all('a'):
    href = anchor.get('href')
    if href and href.endswith('.tar.gz'):
      _links.append(f'{url}{href if not href.startswith("/") else href}')
      _hrefs.append(href)
  
  assert len(_links) == len(_hrefs)

  for i in range(len(_links)):
    dp = os.path.join(root, _hrefs[i][:-7])
    if not os.path.exists(dp):
      links.append(_links[i])
      hrefs.append(_hrefs[i])
    else: print(dp, 'exists')

  for i in trange(len(links)):
    dp = os.path.join(root, hrefs[i][:-7])
    if not os.path.exists(dp):
      rep = requests.get(links[i], stream=True)
      file = tarfile.open(fileobj=rep.raw, mode="r|gz")
      os.mkdir(dp)
      file.extractall(path=dp)

if __name__ == '__main__':
  download_HURSAT_zip('https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2016/')
