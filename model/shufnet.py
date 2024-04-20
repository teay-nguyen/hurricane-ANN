#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init

def conv3x3(device, in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
  return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=stride,
    padding=padding,
    bias=bias,
    groups=groups
  ).to(device)


def conv1x1(device, in_channels, out_channels, groups=1):
  return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=1,
    groups=groups,
    stride=1
  ).to(device)

def channel_shuffle(x:torch.Tensor, groups:int) -> torch.Tensor:
  bs, n_channels, h, w = x.data.size()
  channels_per_group = n_channels // groups
  x = x.view(bs, groups, channels_per_group, h, w)
  x = torch.transpose(x, 1, 2).contiguous()
  return x.view(bs, -1, h, w)

class Unit(nn.Module):
  def __init__(self, device, in_channels, out_channels, groups=3, grouped_conv=True, combine='add'):
    super(Unit, self).__init__()
    self.device = device
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.grouped_conv = grouped_conv
    self.combine = combine
    self.groups = groups
    self.bottleneck_channels = self.out_channels // 4

    if combine == 'add':
      self.depthwise_stride = 1
      self._combine_func = self._add
    elif combine == 'concat':
      self.depthwise_stride = 2
      self._combine_func = self._concat
      self.out_channels -= self.in_channels
    else:
      raise ValueError(f'cannot combine tensors with {self.combine}, only add and comcat are supported')
    
    self.first_1x1_groups = self.groups if grouped_conv else 1
    self.g_conv_1x1_compress = self._make_grouped_conv1x1(
      self.in_channels,
      self.bottleneck_channels,
      self.first_1x1_groups,
      batch_norm=True,
      relu=True)

    self.depthwise_conv3x3 = conv3x3(
      self.device,
      self.bottleneck_channels, self.bottleneck_channels,
      stride=self.depthwise_stride, groups=self.bottleneck_channels)
    self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

    self.g_conv_1x1_expand = self._make_grouped_conv1x1(
      self.bottleneck_channels,
      self.out_channels,
      self.groups,
      batch_norm=True,
      relu=False)

  @staticmethod
  def _add(x, out):
    return x + out

  @staticmethod
  def _concat(x, out):
    return torch.cat((x, out), 1)

  def _make_grouped_conv1x1(self, in_channels, out_channels, groups, batch_norm=True, relu=False):
    modules = OrderedDict()
    conv = conv1x1(self.device, in_channels, out_channels, groups=groups)
    modules['conv1x1'] = conv
    if batch_norm: modules['batch_norm'] = nn.BatchNorm2d(out_channels)
    if relu: modules['relu'] = nn.ReLU()
    if len(modules) > 1: return nn.Sequential(modules)
    else: return conv

  def forward(self, x):
    residual = x
    if self.combine == 'concat':
      residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)
    x = self.g_conv_1x1_compress(x)
    x = channel_shuffle(x, self.groups)
    x = self.depthwise_conv3x3(x)
    x = self.bn_after_depthwise(x)
    x = self.g_conv_1x1_expand(x)
    x = self._combine_func(residual, x)
    return F.relu(x)

class Net(nn.Module):
  def __init__(self, device, in_channels=3, n_classes=2, groups=3):
    super(Net, self).__init__()
    self.device = device
    self.groups = groups
    self.stage_repeats = (3, 7, 3)
    self.in_channels = in_channels
    self.n_classes = n_classes
    if groups == 1: self.stage_out_channels = [-1, 24, 144, 288, 567]
    elif groups == 2: self.stage_out_channels = [-1, 24, 200, 400, 800]
    elif groups == 3: self.stage_out_channels = [-1, 24, 240, 480, 960]
    elif groups == 4: self.stage_out_channels = [-1, 24, 272, 544, 1088]
    elif groups == 8: self.stage_out_channels = [-1, 24, 384, 768, 1536]
    else: raise ValueError(f"{groups} groups not supported for 1x1 grouped convolutions")
    self.conv1 = conv3x3(self.device, self.in_channels, self.stage_out_channels[1], stride=2)
    self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.stage2 = self._make_stage(2)
    self.stage3 = self._make_stage(3)
    self.stage4 = self._make_stage(4)
    n_inputs = self.stage_out_channels[-1]
    self.fc = nn.Linear(n_inputs, self.n_classes).to(device)
    self.init_params()

  def init_params(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
          init.constant_(m.bias, 0)

  def _make_stage(self, stage):
    modules = OrderedDict()
    stage_name = f"Unit Stage {stage}"
    grouped_conv = stage > 2
    first_module = Unit(
      self.device,
      self.stage_out_channels[stage-1],
      self.stage_out_channels[stage],
      groups=self.groups,
      grouped_conv=grouped_conv,
      combine='concat').to(self.device)
    modules[stage_name+'_0'] = first_module
    for i in range(self.stage_repeats[stage-2]):
      name = stage_name+f'_{i+1}'
      module = Unit(
        self.device,
        self.stage_out_channels[stage],
        self.stage_out_channels[stage],
        groups=self.groups,
        grouped_conv=True,
        combine='add').to(self.device)
      modules[name] = module
    return nn.Sequential(modules)

  def forward(self, x):
    x = self.conv1(x)
    x = self.mpool(x)

    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)

    x = F.avg_pool2d(x, x.data.size()[-2:])
    
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = F.sigmoid(x)
    return x
