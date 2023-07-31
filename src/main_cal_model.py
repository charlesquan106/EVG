from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

# from datasets.sample.ctdet_gaze import CTDet_gazeDataset
from datasets.dataset.mpiifacegaze import MpiiFaceGaze
import cv2
import matplotlib.pyplot as plt

from thop import profile
from thop import clever_format


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  # print(opt)
  
  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  
  
  input = torch.randn(1, 3, 256, 256)
  flops, params = profile(model, inputs=(input, ))
  print(flops, params) # ResNet-18_512 14797373440.0 12883478.0
  flops, params = clever_format([flops, params], "%.3f")
  print(flops, params) # ResNet-18_512 14.797G 12.883M

  
  

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)