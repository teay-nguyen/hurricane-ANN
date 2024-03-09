#!/usr/bin/env python3

import numpy as np
import os

def download_HURSAT_zip(url: str):
  root = "../data/HURSAT/"
  from bs4 import BeautifulSoup
  import requests, tarfile
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

def dist_travelled(dir_path: str):
  import math
  from tqdm import trange
  from netCDF4 import Dataset
  import matplotlib.pyplot as plt
  lat, lon, htime = [], [], []
  f_list = os.listdir(dir_path)

  for i in trange(len(f_list)):
    fp = os.path.join(dir_path, f_list[i])
    dat = Dataset(fp)
    lat.append(dat.variables['lat'][:])
    lon.append(dat.variables['lon'][:])
    htime.append(dat.variables['htime'][:])

  lat = np.array(lat, dtype=np.float32).flatten()
  lon = np.array(lon, dtype=np.float32).flatten()
  htime = np.array(htime, dtype=np.float32).flatten()
  dt = lat.shape[0] // htime.shape[0]

  x_pts, y_pts = [], []
  for t in range(len(htime)):
    i0, i1 = t*dt, (t+1)*dt
    if i1 < lat.shape[0]:
      # lo0, lo1, la0, la1 = map(math.radians, [lon[i0], lon[i1], lat[i0], lat[i1]])
      lo0, lo1, la0, la1 = lon[i0], lon[i1], lat[i0], lat[i1]
      d_lon, d_lat = lo1 - lo0, la1 - la0

      a = math.sin(d_lat/2)**2 + math.cos(la0) * math.cos(la1) * math.sin(d_lon/2)**2
      c = 2 * math.asin(math.sqrt(a))
      dst = c * 6371.8

      x_pts.append(htime[t])
      y_pts.append(dst)


  plt.scatter(x_pts, y_pts, s=5)
  plt.show()

if __name__ == '__main__':
  download_HURSAT_zip('https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2016/')