#!/usr/bin/env python3

from bs4 import BeautifulSoup
from tqdm import trange, tqdm
from collections import OrderedDict
from pyproj import Geod
from joblib import Parallel, delayed
from global_land_mask import globe
from pathlib import Path
from typing import List

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
  urls = ['https://www.nhc.noaa.gov/verification/errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt',
          'https://www.nhc.noaa.gov/verification/errors/1989-present_OFCL_v_BCD5_ind_ATL_TI_errors.txt',
          'https://www.nhc.noaa.gov/verification/errors/1989-present_OFCL_v_BCD5_ind_EPAC_TI_errors.txt',
          'https://www.nhc.noaa.gov/verification/errors/1970-present_OFCL_v_BCD5_ind_ATL_AC_errors_noTDs.txt',
          'https://www.nhc.noaa.gov/verification/errors/1989-present_OFCL_v_BCD5_ind_ATL_AC_errors.txt',
          'https://www.nhc.noaa.gov/verification/errors/1989-present_OFCL_v_BCD5_ind_EPAC_AC_errors.txt']

def HURDAT_to_csv(fp: str) -> None:
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

  src_df = pd.DataFrame(hurdat)

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
  save_to = f"{fp.split('/')[-1][:-4]}.csv"
  save_to = os.path.join(SAVE_ROOT, save_to)

  final_df.to_csv(save_to, index=False)

def convert_all_HURDAT_to_csv() -> None:
  files = os.listdir('../data/HURDAT')
  for i in (t := trange(len(files))):
    fp = f'../data/HURDAT/{files[i]}'
    t.set_description(fp)
    HURDAT_to_csv(fp)

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

class HURDAT2(Dataset):
  def __init__(self, hurdat_table, input_vars: List[str], target_vars: List[str], grouping_var: str, time_idx: str, past_horizon: int = 1, future_horizon: int = 1):
    self.hurdat_table = None
    if isinstance(hurdat_table, pd.DataFrame):
        self.hurdat_table = hurdat_table
    if isinstance(hurdat_table, str):
        hurdat_table_path = Path(hurdat_table)
        self.hurdat_table = pd.read_csv(hurdat_table_path)

    self.input_vars = input_vars
    self.target_vars = target_vars

    self.past_horizon = past_horizon
    self.future_horizon = future_horizon

    self.grouping_var = grouping_var
    self.time_idx = time_idx

    self.generated_samples = self.generate_all_ts_samples()

  def generate_all_ts_samples(self) -> List[dict]:
    samps = []
    for actf_code, hurricane_df in tqdm(self.hurdat_table.groupby(self.grouping_var, sort=False)):
      storm_samps = self.generate_storm_ts_samples(hurricane_df)
      samps.extend(storm_samps)
    return samps

  def generate_storm_ts_samples(self, storm_df) -> list[dict]:
    data = []
    for w in storm_df.rolling(window=self.past_horizon+self.future_horizon):
      if w.shape[0] == (self.past_horizon+self.future_horizon):
        data_window = {}

        data_window["input"] = torch.tensor(w.head(self.past_horizon)[self.input_vars].values, dtype=torch.float)
        data_window["input_time_idx"] = torch.tensor(w.head(self.past_horizon)[self.time_idx].values, dtype=torch.float)

        data_window["output"] = torch.tensor(w.tail(self.future_horizon)[self.target_vars].values, dtype=torch.float)
        data_window["output_time_idx"] = torch.tensor(w.tail(self.future_horizon)[self.time_idx].values, dtype=torch.float)

        data_window["window_time_idx"] = torch.tensor(w[self.time_idx].values, dtype=torch.float)
        data_window["atcf_code"] = w['atcf_code'].iloc[0]

        data.append(data_window)
    return data

  def __len__(self):
    return len(self.generated_samples)

  def __getitem__(self, idx):
    return self.generated_samples[idx]

class Hurricane:
  def __init__(self):
    pass

if __name__ == '__main__':
  #fetch_HURSAT('https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2016/')
  #fetch_HURDAT()
  convert_all_HURDAT_to_csv()