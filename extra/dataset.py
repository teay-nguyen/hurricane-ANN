#!/usr/bin/env python

from bs4 import BeautifulSoup
from tqdm import trange, tqdm
from collections import OrderedDict
from pyproj import Geod
from joblib import Parallel, delayed
from global_land_mask import globe
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List
from pprint import pprint

import os
import torch
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, tarfile, datetime, urllib.request, math

from torch.utils.data import Dataset

def fetch_HURSAT(URL: str) -> None:
  root = "../data/HURSAT/"

  rep = requests.get(URL)
  soup = BeautifulSoup(rep.content, 'html.parser')

  _links, _hrefs, links, hrefs = [], [], [], []
  for anchor in soup.find_all('a'):
    href = anchor.get('href')
    if href and href.endswith('.tar.gz'):
      _links.append(f'{URL}{href if not href.startswith("/") else href}')
      _hrefs.append(href)
  
  assert len(_links) == len(_hrefs)

  for i in range(len(_links)):
    dp = os.path.join(root, _hrefs[i][:-7])
    if not os.path.exists(dp):
      links.append(_links[i])
      hrefs.append(_hrefs[i])
    else: print(dp, 'exists')

  if len(links) > 0:
    for i in trange(len(links)):
      dp = os.path.join(root, hrefs[i][:-7])
      if not os.path.exists(dp):
        rep = requests.get(links[i], stream=True)
        file = tarfile.open(fileobj=rep.raw, mode="r|gz")
        os.mkdir(dp)
        file.extractall(path=dp)

def fetch_HURDAT() -> None:
  class Progress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
      if tsize is not None:
        self.total = tsize
      self.update(b * bsize - self.n)

  ROOT = '../data/HURDAT'
  URL = 'https://www.nhc.noaa.gov/data/hurdat/'

  page = requests.get(URL)
  soup = BeautifulSoup(page.content, 'html.parser')

  for anchor in soup.find_all('a'):
    href = anchor.get('href')
    if href and href.endswith('.txt') and href.startswith('hurdat2'):
      with Progress(unit='B', unit_scale=True, miniters=1) as t:
        print('fetching', f'{URL}{href}')
        urllib.request.urlretrieve(f'{URL}{href}', os.path.join(ROOT, href), reporthook=t.update_to)

def fetch_NHC_forecast_error_database() -> None:
  class Progress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
      if tsize is not None:
        self.total = tsize
      self.update(b * bsize - self.n)

  urls = ['https://www.nhc.noaa.gov/verification/errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt',
          'https://www.nhc.noaa.gov/verification/errors/1989-present_OFCL_v_BCD5_ind_ATL_TI_errors.txt',
          'https://www.nhc.noaa.gov/verification/errors/1989-present_OFCL_v_BCD5_ind_EPAC_TI_errors.txt',
          'https://www.nhc.noaa.gov/verification/errors/1970-present_OFCL_v_BCD5_ind_ATL_AC_errors_noTDs.txt',
          'https://www.nhc.noaa.gov/verification/errors/1989-present_OFCL_v_BCD5_ind_ATL_AC_errors.txt',
          'https://www.nhc.noaa.gov/verification/errors/1989-present_OFCL_v_BCD5_ind_EPAC_AC_errors.txt']

  ROOT = '../data/NHC_FORECAST_ERROR'
  for url in urls:
    nm = url.split('/')[-1]
    fp = os.path.join(ROOT, nm)
    with Progress(unit='B', unit_scale=True, miniters=1) as t:
      print('fetching', url)
      urllib.request.urlretrieve(url, fp, reporthook=t.update_to)

def HURDAT_to_df(fp: str) -> pd.DataFrame:
  hurdat, lines = [], []

  with open(fp, 'r') as f:
    lines = f.read().splitlines()

  current_atcf_code = None
  current_storm_name = None
  current_storm_n_entries = None

  for line in lines:
    line_dat = line.split(',')[:-1]
    line_dat = list(map(str.strip, line_dat))
    if len(line_dat) == 3:
      current_atcf_code = line_dat[0]
      current_storm_name = line_dat[1]
      current_storm_n_entries = line_dat[2]
    else:
      data = OrderedDict()
      data['atcf_code'] = current_atcf_code
      data['storm_name'] = current_storm_name
      data['year'] = int(line_dat[0][:4])
      data['month'] = int(line_dat[0][4:6])
      data['day'] = int(line_dat[0][6:])
      data['hour'] = int(line_dat[1][:2])
      data['minute'] = int(line_dat[1][2:])
      data['record_id'] = line_dat[2]
      data['system_status'] = line_dat[3]
      lat = float(line_dat[4][:-1])

      if line_dat[4][-1] == 'S': lat *= -1.0
      data['latitude'] = lat

      lon = float(line_dat[5][:-1])
      if(line_dat[5][-1] == 'W'): lon *= -1.0
      data['longitude'] = lon

      data['max_sus_wind'] = float(line_dat[6])
      data['min_pressure'] = float(line_dat[7])

      data['wind_radii_34_NE'] = float(line_dat[8])
      data['wind_radii_34_SE'] = float(line_dat[9])
      data['wind_radii_34_SW'] = float(line_dat[10])
      data['wind_radii_34_NW'] = float(line_dat[11])

      data['wind_radii_50_NE'] = float(line_dat[12])
      data['wind_radii_50_SE'] = float(line_dat[13])
      data['wind_radii_50_SW'] = float(line_dat[14])
      data['wind_radii_50_NW'] = float(line_dat[15])

      data['wind_radii_64_NE'] = float(line_dat[16])
      data['wind_radii_64_SE'] = float(line_dat[17])
      data['wind_radii_64_SW'] = float(line_dat[18])
      data['wind_radii_64_NW'] = float(line_dat[19])

      hurdat.append(data)
  return pd.DataFrame(hurdat)

def HURDAT2_to_df(fp: str) -> pd.DataFrame:
  db = []
  with open(fp, 'r') as f:
    for line in f:
      line = line.replace(' ', '').split(',')
      if line[0][:2] == 'AL':
        storm_id = line[0]
        storm_name = line[1]
        storm_entries = line[2]
        for i in range(int(storm_entries)):
          ent = f.readline().replace(' ', '').split(',')
          ent = [None if x == '-999' else x for x in ent]
          timestamp = datetime.datetime(int(ent[0][:4]), int(ent[0][4:6]), int(ent[0][6:8]), int(ent[1][:2]), int(ent[1][3:]))
          db.append([storm_id, storm_name, timestamp] + ent[2:-1])
      else: print('unidentified storm '.join(str(line[0])))

  return pd.DataFrame(db, columns = ['storm_id', 'storm_name', 'entry_time', 'entry_id', 'entry_status',
                                     'lat', 'long','max_wind', 'min_pressure', '34kt_ne', '34kt_se',
                                     '34kt_sw', '34kt_nw', '50kt_ne', '50kt_se', '50kt_sw',
                                     '50kt_nw', '64kt_ne', '64kt_se', '64kt_sw', '64kt_nw'])

def HURDAT_to_csv(fp: str) -> None:
  src_df = HURDAT_to_df(fp)

  hurricane_df_list = []

  for atcf_code, hurricane_df in src_df.groupby('atcf_code', sort=False):
    hurricane_df_list.append(hurricane_df)

  def hurricane_values_missing_filter(x):
    flag = True
    if -999 in x['max_sus_wind'].values: flag = False
    if -999 in x['min_pressure'].values: flag = False
    return flag

  def odd_time_row_filter(x):
    ret = x[((x['hour'] % 6) == 0) & (x['minute'] == 0)]
    return ret

  def calculate_delta_distance_and_azimuth(x):
    x['latitude-6'] = x['latitude'].shift(1)
    x['longitude-6'] = x['longitude'].shift(1)
    x.dropna(inplace=True)

    wgs84_geod = Geod(ellps='WGS84')
    def delta_distance_azimuth(lat1,lon1,lat2,lon2):
      az12, az21, dist = wgs84_geod.inv(lon1,lat1,lon2,lat2)
      dist = [x / 1000.0 for x in dist]
      return dist, az12

    x['delta_distance'], x['azimuth'] = delta_distance_azimuth(x['latitude-6'].tolist(),x['longitude-6'].tolist(),x['latitude'].tolist(),x['longitude'].tolist())
    x['delta_distance_x'] = np.sin(np.deg2rad(x['azimuth']))*x['delta_distance']
    x['delta_distance_y'] = np.cos(np.deg2rad(x['azimuth']))*x['delta_distance']
    
    del x['latitude-6']
    del x['longitude-6']
    return x

  def calculate_x_y(x):
    x['x'] = np.sin(x['latitude']) * np.cos(x['longitude'])
    x['y'] = np.sin(x['latitude']) * np.sin(x['longitude'])
    return x

  def calculate_new_dt_info(x):
    def extract_dt_info(y):
      y['day_of_year'] = datetime.datetime(y['year'],y['month'],y['day']).timetuple().tm_yday
      y['minute_of_day'] = y['hour']*60 + y['minute']
      return y
    x = x.apply(extract_dt_info, axis=1)
    return x

  def calculate_jday(x):
    x['jday'] = np.abs(x['day_of_year']-253)
    return x

  def calculate_vpre(x):
    x['vpre'] = x['max_sus_wind'] * x['min_pressure']
    x['vpre_inverse_scaled'] = (x['max_sus_wind'] * x['min_pressure']) / (x['max_sus_wind'] + x['min_pressure'])
    return x

  def calculate_landfall(x):
    x['landfall'] = globe.is_land(x['latitude'], x['longitude'])
    x['landfall'] = x['landfall'].astype(int)
    return x

  def create_time_idx(x):
    x = x.sort_values(by=['year', 'month', 'day', 'hour'])
    x['time_idx'] = np.arange(len(x.index))
    return x

  '''cleaning and feature extraction'''
  hurricane_df_list = list(filter(hurricane_values_missing_filter, hurricane_df_list))
  hurricane_df_list = list(map(odd_time_row_filter, hurricane_df_list))
  hurricane_df_list = Parallel(n_jobs=-1,verbose=0)(delayed(calculate_delta_distance_and_azimuth)(h_df) for h_df in hurricane_df_list)
  hurricane_df_list = Parallel(n_jobs=-1,verbose=0)(delayed(calculate_x_y)(h_df) for h_df in hurricane_df_list)
  hurricane_df_list = Parallel(n_jobs=-1,verbose=0)(delayed(calculate_new_dt_info)(h_df) for h_df in hurricane_df_list)
  hurricane_df_list = Parallel(n_jobs=-1,verbose=0)(delayed(calculate_jday)(h_df) for h_df in hurricane_df_list)
  hurricane_df_list = Parallel(n_jobs=-1,verbose=0)(delayed(calculate_vpre)(h_df) for h_df in hurricane_df_list)
  # hurricane_df_list = list(map(calculate_landfall, hurricane_df_list))
  hurricane_df_list = Parallel(n_jobs=-1,verbose=0)(delayed(calculate_landfall)(h_df) for h_df in hurricane_df_list)

  # print("Calculating time shifted feature...")
  # hurricane_df_list = list(map(calculate_time_shifted_features, hurricane_df_list))
  # hurricane_df_list = Parallel(n_jobs=-1,verbose=0)(delayed(calculate_time_shifted_features)(h_df) for h_df in hurricane_df_list)

  hurricane_df_list = Parallel(n_jobs=-1, verbose=0)(delayed(create_time_idx)(h_df) for h_df in hurricane_df_list)
  hurricane_df_list = [h_df for h_df in hurricane_df_list if not h_df.empty]
  hurricane_df_list = [h_df for h_df in hurricane_df_list if np.all(h_df['year'] >= 1982)]
  hurricane_df_list = [h_df for h_df in hurricane_df_list if np.any(h_df['system_status'] == 'HU') or np.any(h_df['system_status'] == 'TS')]

  final_df = pd.concat(hurricane_df_list)
  final_df = final_df.sort_values(by=['year','atcf_code', 'month','day', 'hour'])

  SAVE_ROOT = "../data/HURDAT_CSV"
  save_to = os.path.join(SAVE_ROOT, f"{fp.split('/')[-1][:-4]}.csv")
  final_df.to_csv(save_to, index=False)

def convert_all_HURDAT_to_csv() -> None:
  files = os.listdir('../data/HURDAT')
  for i in (t := trange(len(files))):
    fp = f'../data/HURDAT/{files[i]}'
    t.set_description(fp)
    try: HURDAT_to_csv(fp)
    except: print(f"can't process {fp}")

@dataclass
class ModelObj:
  name: str
  storm: dict
  def __init__(self, nm):
    self.name = nm
    self.storm = {}

def parse_NHC_forecast_error(fp:str='../data/NHC_FORECAST_ERROR/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt') -> dict:
  models = {}
  with open(fp, 'r') as f:
    lines = f.read().splitlines()
    line = lines[1].split()
    names = line[2:]
    for name in names:
      models[name] = ModelObj(name)
    for line in lines[9:]:
      line = line.split()
      timestamp = datetime.datetime.strptime(line[0], '%d-%m-%Y/%H:%M:%S')
      storm_id = line[1]
      samp_sizes = {"F012": float(line[2]), "F024": float(line[3]),"F036": float(line[4]), "F048": float(line[5]), "F072": float(line[6]), "F096": float(line[7]), "F120": float(line[8]), "F144": float(line[9]), "F168": float(line[10])}
      latitude = float(line[11])
      longitude = float(line[12])
      wind_spd = float(line[13])

      for i in range(len(names)):
        intensity_forecast = dict(list(zip([timestamp, timestamp + timedelta(hours = 12), timestamp + timedelta(hours = 24), timestamp + timedelta(hours = 36), timestamp + timedelta(hours = 48), timestamp + timedelta(hours = 72), timestamp + timedelta(hours = 96), timestamp + timedelta(hours = 120), timestamp + timedelta(hours = 144), timestamp + timedelta(hours = 168)], [None if x == "-9999.0" else float(x) for x in line[14 + (20 * i) : 24 + (20 * i)]])))
        track_forecast = dict(list(zip([timestamp, timestamp + timedelta(hours = 12), timestamp + timedelta(hours = 24), timestamp + timedelta(hours = 36), timestamp + timedelta(hours = 48), timestamp + timedelta(hours = 72), timestamp + timedelta(hours = 96), timestamp + timedelta(hours = 120), timestamp + timedelta(hours = 144), timestamp + timedelta(hours = 168)], [None if x == "-9999.0" else float(x) for x in line[24 + (20 * i) : 34 + (20 * i)]])))
        if storm_id not in models[names[i]].storm.keys():
          models[names[i]].storm[storm_id] = dict()
        models[names[i]].storm[storm_id].update({
          timestamp: {
            "sample_sizes": samp_sizes,
            "lat": latitude,
            "long": longitude,
            "wind_speed": wind_spd,
            "intensity_forecast": intensity_forecast,
            "track_forecast": track_forecast
          }
        })

  return models

def dist_travelled_latitude_longitude(dir_path: str) -> None:
  lat, lon, htime = [], [], []
  f_list = os.listdir(dir_path)

  for i in trange(len(f_list)):
    fp = os.path.join(dir_path, f_list[i])
    dat = netCDF4.Dataset(fp)
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
      lo0, lo1, la0, la1 = map(math.radians, [lon[i0], lon[i1], lat[i0], lat[i1]])
      d_lon, d_lat = lo1 - lo0, la1 - la0
      a = math.sin(d_lat/2)**2 + math.cos(la0) * math.cos(la1) * math.sin(d_lon/2)**2
      c = 2 * math.asin(math.sqrt(a))
      dst = c * 6371.8
      x_pts.append(htime[t])
      y_pts.append(dst)

  plt.scatter(x_pts, y_pts, s=5)
  plt.show()

class Hurricane:
  def __init__(self, name: str, id: str):
    self.name = name
    self.id = id
    self.entries = dict()
    self.models = dict()

  def add_entry(self, arr):
    ent = {
      arr[0] : {
        'entry_time': arr[0],
        'entry_id' : arr[1],
        'entry_status': arr[2],
        'lat' : float(arr[3][:-1]),
        'long': float(arr[4][:-1]),
        'max_wind': float(arr[5]),
        'min_pressure': None if arr[6] is None else float(arr[6]),
        'wind_radii': arr[7:]
      }
    }

    self.entries.update(ent)

  def add_model(self, name, model):
    self.models[name] = model

if __name__ == '__main__':
  #fetch_HURSAT('https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2016/')
  #fetch_HURDAT()
  #convert_all_HURDAT_to_csv()
  #dat_df = HURDAT_to_df('../data/HURDAT/hurdat2-1851-2017-050118.txt')
  #fetch_NHC_forecast_error_database()
  df = HURDAT2_to_df('../data/HURDAT/hurdat2-atl-02052024.txt')
  err = parse_NHC_forecast_error('../data/NHC_FORECAST_ERROR/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt')
  print(df.query('storm_id == "AL122005"').head())
  pprint(err['OFCL'].storm['AL122005'][datetime.datetime(2005,8,28,18,0)], indent=8)
