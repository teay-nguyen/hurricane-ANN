#!/usr/bin/env python

from shutil import copy2 as cp
from typing import Tuple, Optional, List, Dict, Union, Callable
from joblib import Parallel, delayed
from shapely.wkt import loads
from torchvision.transforms import v2
from torch.utils.data import Dataset
import cv2, json, os, glob, tqdm
import numpy as np
import numpy.typing as npt
import torch

np.random.seed(1337)
os.environ['OMP_NUM_THREADS'] = '1'

torch.manual_seed(1337)
torch.set_num_threads(1)

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

''' xBD dataset '''

DAMAGE_SUBTYPES = {
  "no-damage": 1,
  "minor-damage": 2,
  "major-damage": 3,
  "destroyed": 4,
  "un-classified": 1
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

'''
Building Damage Annotation on Post-Hurricane
Satellite Imagery Based on Convolutional Neural
Networks (Quoc Dung Cao & Youngjun Choe)
'''

def get_hurricane_files(subset:str) -> Dict[str, List[str]]:
  assert subset in ('test', 'test_another', 'train_another', 'validation_another')
  ret = {'damage':[], 'no_damage':[]}
  root = '../data/images/Post-hurricane'
  p_subset = os.path.join(root, subset)
  ret['damage'] = [os.path.join(p_subset, 'damage', fp) for fp in os.listdir(os.path.join(p_subset, 'damage'))]
  ret['no_damage'] = [os.path.join(p_subset, 'no_damage', fp) for fp in os.listdir(os.path.join(p_subset, 'no_damage'))]
  return ret

def load_hurricane_imgs() -> Dict[str, Dict[str, npt.NDArray[np.uint8]]]:
  from PIL import Image
  os.makedirs('./hurricane_npy', exist_ok=True)
  datasets = {'test':dict(), 'test_another':dict(), 'train_another':dict(), 'validation_another':dict()}
  for subset in datasets:
    k0, k1 = f'./hurricane_npy/{subset}_damage.npy', f'./hurricane_npy/{subset}_no_damage.npy'
    files = get_hurricane_files(subset)
    if not os.path.exists(k0):
      print('populating', k0)
      datasets[subset]['damage'] = np.array([np.array(Image.open(files['damage'][i])) for i in tqdm.trange(len(files['damage']))], dtype=np.uint8)
      np.save(k0, datasets[subset]['damage'])
    else: datasets[subset]['damage'] = np.load(k0).astype(np.uint8)
    if not os.path.exists(k1):
      print('populating', k1)
      datasets[subset]['no_damage'] = np.array([np.array(Image.open(files['no_damage'][i])) for i in tqdm.trange(len(files['no_damage']))], dtype=np.uint8)
      np.save(k1, datasets[subset]['no_damage'])
    else: datasets[subset]['no_damage'] = np.load(k1).astype(np.uint8)
  return datasets

def generate_labels(dataset:Dict[str, Dict[str, npt.NDArray[np.uint8]]], subset:str='train_another', shuffle:bool=False, to_tensor:bool=False) -> \
                    Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]] | Tuple[torch.Tensor, torch.Tensor]:
  assert subset in dataset, 'subset is not valid'
  X = np.concatenate((dataset[subset]['damage'], dataset[subset]['no_damage']), axis=0).astype(np.uint8)
  Y = np.array(([1]*dataset[subset]['damage'].shape[0])+([0]*dataset[subset]['no_damage'].shape[0]), dtype=np.uint8)
  if shuffle: shuf = np.random.permutation(X.shape[0]); X, Y = X[shuf], Y[shuf]
  if to_tensor: return torch.tensor(X), torch.tensor(Y)
  else: return X, Y

def shuffle_data(X:Union[npt.NDArray[np.uint8], torch.Tensor], Y:Union[npt.NDArray[np.uint8], torch.Tensor]) -> \
                 Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]] | Tuple[torch.Tensor, torch.Tensor]:
  assert type(X) == type(Y), 'X is not the same as Y'
  if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray): perm = np.random.permutation(X.shape[0])
  else: perm = torch.randint(0, X.shape[0], (X.shape[0],))
  return X[perm], Y[perm]

def upscale_img(X:Union[npt.NDArray[np.uint8], torch.Tensor]) -> torch.Tensor:
  return v2.Resize((150,150))(X)

def scale_and_upscale_img(X:Union[npt.NDArray[np.uint8], torch.Tensor]) -> torch.Tensor:
  if isinstance(X, np.ndarray):
    return v2.Resize((150,150))(torch.tensor(X).float() / 255.)
  return v2.Resize((150,150))(X.float() / 255.)

def permute_img_for_train(img:torch.Tensor) -> torch.Tensor:
  assert len(img.shape) == 4 and img.shape[-1] == 3
  return img.permute(0, 3, 1, 2)

def permute_img_for_view(img:torch.Tensor) -> torch.Tensor:
  assert len(img.shape) == 4 and img.shape[1] == 3
  return img.permute(0, 2, 3, 1)

def generate_augmented_imgs(X:Union[npt.NDArray[np.uint8], npt.NDArray[np.float32], torch.Tensor],
                            Y:Union[npt.NDArray[np.uint8], npt.NDArray[np.float32], torch.Tensor],
                            loadpath_X:str, loadpath_Y:str, scale=True) -> None:
  # this is the dumbest shit i've written so far
  # TODO: find more efficient way to do this
  if isinstance(X, np.ndarray): X = torch.tensor(X)
  if isinstance(Y, np.ndarray): Y = torch.tensor(Y)
  if scale: X = (X.float() / 255.)
  X = upscale_img(X.permute(0,3,1,2))
  composes = (
    (v2.Compose([v2.RandomHorizontalFlip(p=1), v2.Resize((150,150))]), 'horizontal_flip'),
    (v2.Compose([v2.RandomVerticalFlip(p=1), v2.Resize((150,150))]), 'vertical_flip'),
    (v2.Compose([v2.RandomAffine(degrees=(0,0), translate=[0,.2], shear=[-10,10,-10,10]),
                 v2.Resize((150,150))]), 'random_affine'),
    (v2.Compose([v2.RandomRotation((-100, 100)),
                 v2.Resize((150,150))]), 'rotation'),
    (v2.Compose([v2.RandomHorizontalFlip(p=1),
                 v2.RandomVerticalFlip(p=1),
                 v2.Resize((150,150))]), 'flip_hv_transform'),
    (v2.Compose([v2.ColorJitter(brightness=.2, contrast=.1, saturation=.2, hue=.3),
                 v2.Resize((150,150))]), 'color_jitter'),
    (v2.Compose([v2.RandomResizedCrop(size=(128,128)),
                 v2.Resize((150,150))]), 'random_resized_crop'),
    (v2.Compose([v2.RandomAdjustSharpness(sharpness_factor=2),
                 v2.Resize((150,150))]), 'random_adjust_sharpness'))
  X_a, Y_a = [], []
  print(f'{bcolors.BOLD}[info]{bcolors.ENDC} generating augmented images...')
  for i in range(len(composes)):
    print(f'   {bcolors.BOLD}[info]{bcolors.ENDC} applying transform: {composes[i][1]}')
    samps = torch.randint(0, X.shape[0], (5000,))
    X_a.append(composes[i][0](X[samps]))
    Y_a.append(Y[samps])
  print(f'{bcolors.BOLD}[info]{bcolors.ENDC} merging augmented images...')
  X_a, Y_a = torch.cat(X_a, dim=0), torch.cat(Y_a, dim=0)
  ret_X = torch.empty(X.shape[0]+X_a.shape[0], 3, 150, 150)
  ret_Y = torch.empty(Y.shape[0]+Y_a.shape[0])
  torch.cat([X, X_a], dim=0, out=ret_X)
  torch.cat([Y, Y_a], dim=0, out=ret_Y)
  ret_X = ret_X.permute(0, 2, 3, 1)
  print(f'{bcolors.BOLD}[info]{bcolors.ENDC} saving augmented images to {loadpath_X}')
  np.save(loadpath_X, ret_X.detach().numpy().astype(np.float32))
  print(f'{bcolors.BOLD}[info]{bcolors.ENDC} saving augmented images to {loadpath_Y}')
  np.save(loadpath_Y, ret_Y.detach().numpy().astype(np.uint8))
  print(f'{bcolors.BOLD}[info]{bcolors.ENDC} augmented images saved to {loadpath_X} and {loadpath_Y}')

def fetch_label_batch(batch_idx:npt.NDArray[np.int32], loadpath_X:str, loadpath_Y:str, to_tensor=False) ->\
                      Union[Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]], Tuple[torch.Tensor, torch.Tensor]]:
  X_train = np.load(loadpath_X, mmap_mode='r')[batch_idx]
  Y_train = np.load(loadpath_Y, mmap_mode='r')[batch_idx]
  if not to_tensor: return X_train.astype(np.float32), Y_train.astype(np.uint8)
  else: return torch.tensor(X_train), torch.tensor(Y_train)

class HurricaneImages(Dataset):
  def __init__(self, X:Union[npt.NDArray[np.uint8], torch.Tensor], Y:Union[npt.NDArray[np.uint8], torch.Tensor],
               transform:Optional[Callable[[torch.Tensor], torch.Tensor]]=None):
    self.X = torch.tensor(X) if isinstance(X, np.ndarray) else X
    self.Y = torch.tensor(Y) if isinstance(Y, np.ndarray) else Y
    self.transform = transform

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, i):
    if self.transform:
      return self.transform(self.X[i].permute(2,0,1).float()), self.Y[i]
    return self.X[i].permute(2,0,1).float(), self.Y[i]

  def __str__(self):
    return f'<{self.__class__.__name__} object at {hex(id(self))}, X={self.X.shape}, Y={self.Y.shape}, transform={self.transform}>'
