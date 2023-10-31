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
import tqdm

# from datasets.sample.ctdet_gaze import CTDet_gazeDataset
from datasets.dataset.mpiifacegaze import MpiiFaceGaze
import cv2
import matplotlib.pyplot as plt
from early_stopping import EarlyStopping


def should_exclude(idx,item):
    # print(f"index: {idx}")
    vp_gazepoint_x, vp_gazepoint_y = item['meta']['vp_gazepoint']

    return vp_gazepoint_x > opt.vp_w or vp_gazepoint_x < 0 or vp_gazepoint_y > opt.vp_h or vp_gazepoint_y < 0


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  # print(opt)
  
  early_stopping = EarlyStopping(patience=4,delta=10)
  logger = Logger(opt)

  force_server_training = 0
  if force_server_training == 0 :
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  else:	
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    torch.cuda.set_device(1)
    
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  
  print('*****************')
  
  for i in range(2):
    index = i

    dataset_test = Dataset(opt, 'train')
    # sample_hm = dataset_test[index]["hm"]
    sample = dataset_test[index]
    # smaple_c = sample["meta"]["c"]
    # smaple_s = sample["meta"]["s"]
    # print(f"c = {smaple_c}")
    # print(f"s = {smaple_s}")
    
    
    # for k,v in sample.items() :
      
    #   key_shape = sample[k].shape
    #   print(f"{k}:{key_shape}")
    sample_input = sample["input"]
    sample_input = sample_input.transpose(1,2,0)
    sample_input = cv2.cvtColor(sample_input, cv2.COLOR_BGR2RGB)
    
    sample_hm = sample["hm"]
    sample_hm = sample_hm.transpose(1,2,0)
    
    # plt.figure(figsize=(14, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(sample_input)
    # plt.title('input image')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(sample_hm, cmap="gray")
    # plt.title('gaze hm')
    # plt.show()
    
    if opt.face_hm_head: 

      sample_face_hm = sample["face_hm"]
      sample_face_hm = sample_face_hm.transpose(1,2,0)
      
      
      # plt.figure(figsize=(14, 4))
      # plt.subplot(1, 2, 1)
      # plt.imshow(sample_input)
      # plt.title('input image')
      
      # plt.subplot(1, 2, 2)
      # plt.imshow(sample_face_hm, cmap="gray")
      # plt.title('face hm')
      # plt.show()
      
    vis_hm = 1
    if vis_hm: 

      sample_hm_d = sample["hm_d"]
      sample_hm_d = sample_hm_d.transpose(1,2,0)
      
      
      plt.figure(figsize=(14, 4))
      plt.subplot(1, 2, 1)
      plt.imshow(sample_hm, cmap="gray")
      plt.title('sample_hm')
      
      plt.subplot(1, 2, 2)
      plt.imshow(sample_hm_d, cmap="gray")
      plt.title('sample_hm_d')
      plt.show()
    
    
  
  print('*****************')
  # exclude_val_index_list = []
  # if not opt.not_data_train_val_exclude :
  #   exclude_val_index_list = [idx for idx, item in enumerate(tqdm.tqdm(Dataset(opt, 'val'))) if should_exclude(idx,item)]
  # print("**********vp_gazepoint over virtual plane range need exclude**********")
  # print(f"exclude_val_index_list: {exclude_val_index_list}")
  # print(f"exclude_val_index_list len: {len(exclude_val_index_list)}")
  # exclude_val_sampler = torch.utils.data.sampler.SubsetRandomSampler([idx for idx in range(len(Dataset(opt, 'val'))) if idx not in exclude_val_index_list])

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return
  # print("**********start check **********")
  # exclude_train_index_list = []
  # if not opt.not_data_train_val_exclude :
  #   exclude_train_index_list = [idx for idx, item in enumerate(tqdm.tqdm(Dataset(opt, 'train'))) if should_exclude(idx,item)]
  # print("**********vp_gazepoint over virtual plane range need exclude**********")
  # print(f"exclude_train_index_list: {exclude_train_index_list}")
  # print(f"exclude_train_index_list len: {len(exclude_train_index_list)}")
  # exclude_train_sampler = torch.utils.data.sampler.SubsetRandomSampler([idx for idx in range(len(Dataset(opt, 'train'))) if idx not in exclude_train_index_list])
  #####    #####       #####
  
  
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  print(f"batch_size: {opt.batch_size}")
  
  # for batch in enumerate(train_loader):
  #   pass
  val_loss = 0
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:.2f} | '.format(k, v))
      

    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
        
        if k == "pog_loss" :
          val_loss = v
          early_stopping(val_loss)
        
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch % opt.val_intervals == 0:
      # every val_intervals epoch will save model weights
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
          
    #达到早停止条件时，early_stop会被置为True
    if early_stopping.early_stop:
        print("Early stopping !!!!")
        break #跳出迭代，结束训练
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)