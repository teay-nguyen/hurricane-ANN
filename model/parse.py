#!/usr/bin/env python

from shutil import copy2 as cp
from typing import Tuple, Optional, List, Dict
from joblib import Parallel, delayed
from shapely.wkt import loads
import cv2, json, os, glob, tqdm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

''' xBD dataset '''

DAMAGE_SUBTYPES = {
  "no-damage": 1,
  "minor-damage": 2,
  "major-damage": 3,
  "destroyed": 4,
  "un-classified": 1 #?
}

def mask_polygon(poly, in_shape:Tuple[int, int]=(1024,1024)) -> npt.NDArray[np.uint8]:
  img_mask = np.zeros(in_shape, dtype=np.uint8)
  int_coords = lambda x: np.array(x).round().astype(np.int32)
  exteriors = [int_coords(poly.exterior.coords)]
  interiors = [int_coords(pi.coords) for pi in poly.interiors]
  cv2.fillPoly(img_mask, exteriors, 1)
  cv2.fillPoly(img_mask, interiors, 0)
  return img_mask

def process_xBD_json(path:str, saveto:str) -> Optional[Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]:
  assert os.path.isdir(f'./masks/{saveto}')
  if not "_pre_disaster" in path: return
  pre = json.load(open(path))
  post = json.load(open(path.replace("_pre_disaster", "_post_disaster")))
  mask_pre = np.zeros((1024,1024), dtype=np.uint8)
  mask_post = np.zeros((1024,1024), dtype=np.uint8)
  for feature in pre['features']['xy']:
    poly = loads(feature['wkt'])
    mask_pre[mask_polygon(poly) > 0] = 255
  for feature in post['features']['xy']:
    poly = loads(feature['wkt'])
    mask_post[mask_polygon(poly) > 0] = DAMAGE_SUBTYPES[feature['properties']['subtype']]
  cv2.imwrite(f'./masks/{saveto}/{os.path.basename(path).replace(".json", ".png")}', mask_pre, [cv2.IMWRITE_PNG_COMPRESSION, 9])
  cv2.imwrite(f'./masks/{saveto}/{os.path.basename(path).replace("_pre_disaster.json", "_post_disaster.png")}', mask_post, [cv2.IMWRITE_PNG_COMPRESSION, 9])
  return mask_pre, mask_post

def process_xBD_jsons(dataset:str) -> None:
  assert dataset in ('train', 'test', 'tier3')
  saveto = {'train':'train_masks', 'test':'test_masks', 'tier3':'tier3_masks'}[dataset]
  os.makedirs(f'./masks/{saveto}', exist_ok=True)
  paths = tuple(filter(lambda x: "pre" in x, glob.glob(f'../data/images/xview2/{dataset}/labels/*')))
  Parallel(n_jobs=-1)(delayed(process_xBD_json)(path, saveto) for path in tqdm.tqdm(paths, total=len(paths)))

def get_xBD_files(subset:str) -> List[str]:
  root = {'train':'train_masks', 'test':'test_masks', 'tier3':'tier3_masks'}[subset]
  root = os.path.join('./masks', root)
  files = [f for f in next(os.walk(root))[2] if 'pre' in f]
  return files

def mv_xBD_files() -> None:
  masks = {'train':get_xBD_files('train'), 'test':get_xBD_files('test'), 'tier3':get_xBD_files('tier3')}
  for i, v in masks.items():
    os.makedirs(os.path.join('./disasters', i), exist_ok=True)
    for fname in v:
      disaster = fname.split('_')[0]
      os.makedirs(os.path.join('./disasters', i, disaster), exist_ok=True)
      cp(os.path.join(f'./masks/{i}_masks', fname), os.path.join(f'./disasters/{i}/{disaster}', fname))
      pfname = fname.replace('_pre_', '_post_')
      cp(os.path.join(f'./masks/{i}_masks', pfname), os.path.join(f'./disasters/{i}/{disaster}', pfname))

def xBD_npy() -> None:
  import cv2, tqdm
  os.makedirs('./disasters_npy', exist_ok=True)
  os.makedirs('./disasters_npy/train', exist_ok=True)
  os.makedirs('./disasters_npy/test', exist_ok=True)
  os.makedirs('./disasters_npy/tier3', exist_ok=True)
  disaster_masks = {'train':'./disasters/train', 'test':'./disasters/test', 'tier3':'./disasters/tier3'}
  for i, disasters in disaster_masks.items():
    for d in os.listdir(disasters):
      dp = os.path.join('./disasters_npy', i, d)
      os.makedirs(dp, exist_ok=True)
      img_dir = os.listdir(os.path.join('./disasters', i, d))
      for img_idx in (t := tqdm.trange(len(img_dir))):
        t.set_description(f'{i} {d} {img_dir[img_idx]}')
        p_img = os.path.join('./disasters', i, d, img_dir[img_idx])
        img = cv2.imread(p_img, cv2.IMREAD_GRAYSCALE)
        np.save(os.path.join('./disasters_npy', i, d, img_dir[img_idx]).replace('.png', '.npy'), img)

''' hurricane dataset '''

def get_hurricane_files(subset:str) -> Dict[str, List[str]]:
  assert subset in ('test', 'test_another', 'train_another', 'validation_another')
  ret = {'damage':[], 'no_damage':[]}
  root = '../data/images/hurricane-imgs'
  p_subset = os.path.join(root, subset)
  ret['damage'] = [os.path.join(p_subset, 'damage', fp) for fp in os.listdir(os.path.join(p_subset, 'damage'))]
  ret['no_damage'] = [os.path.join(p_subset, 'no_damage', fp) for fp in os.listdir(os.path.join(p_subset, 'no_damage'))]
  return ret

if __name__ == "__main__":
  from PIL import Image
  files = get_hurricane_files('train_another')
  img = np.array(Image.open(files['damage'][0]), dtype=np.uint8)
  plt.imshow(img)
  plt.show()
