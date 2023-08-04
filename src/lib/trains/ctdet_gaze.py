from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_gaze_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_gaze_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime
import shutil

class CtdetGazeLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetGazeLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_pog = torch.nn.MSELoss()
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, off_loss, pog_loss = 0, 0, 0
    # print(f"outputs['hm']: {outputs['hm'].shape}")
    # print(f"batch['hm']: {batch['hm'].shape}")
    # out_resnet = outputs[1]
    # outputs = outputs[0]
    
    for s in range(opt.num_stacks):
      output = outputs[s]
      
      
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        print("eval_oracle_hm")
        output['hm'] = batch['hm']
        
      if opt.eval_oracle_offset:
        print("eval_oracle_offset")
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
      
      output_hm = output['hm'][0].detach().cpu().numpy()
      output_hm = output_hm.transpose(1, 2, 0)
      
      image_folder = "/home/owenserver/Python/CenterNet_gaze/hm_image"
      if not os.path.exists(image_folder):
        os.makedirs(image_folder)
      
      src_path = "/home/owenserver/Python/CenterNet_gaze/exp/.gitignore"
      dest_path = "/home/owenserver/Python/CenterNet_gaze/hm_image/.gitignore"
      if not os.path.exists(dest_path):
        shutil.copy(src_path, dest_path)
      

      current_time = datetime.now()
      # 格式化時間為指定的字串格式（例如：%Y%m%d_%H%M%S）
      time_str = current_time.strftime("%Y%m%d_%H%M%S")
      # 構造文件名
      filename_hm = f"{time_str}_hm.jpg"
      
      save_output_hm_path = os.path.join(image_folder,filename_hm)
      if opt.heat_map_debug:
        plt.imsave(save_output_hm_path, output_hm.squeeze(), cmap='gray')

      # fig_output_hm, ax_output_hm = plt.subplots(figsize=[5,5])
      # im_output_hm = ax_output_hm.imshow(output_hm,
      #     origin='lower',
      #     cmap='hot', 
      #     )
      
      # fig_output_hm.colorbar(im_output_hm)
      # plt.axis('off') 
      # plt.savefig(save_output_hm_path)
      # plt.show()
      
      # plt.imshow(output_hm, cmap="gray")
      # plt.axis('off')  
      # plt.show()
      
      
      
      batch_hm = batch["hm"][0].detach().cpu().numpy()
      batch_hm = batch_hm.transpose(1, 2, 0)
      
      filename_gt = f"{time_str}_gt.jpg"
      
      save_gt_hm_path = os.path.join(image_folder,filename_gt)
      if opt.heat_map_debug:
        plt.imsave(save_gt_hm_path, batch_hm.squeeze(), cmap='gray')
      
      # fig_gt_hm, ax_gt_hm = plt.subplots(figsize=[5,5])
      # im_gt_hm = ax_gt_hm.imshow(batch_hm,
      #     origin='lower',
      #     cmap='hot', 
      #     )
      # 
      # fig_gt_hm.colorbar(im_gt_hm)
      # plt.axis('off') 
      # plt.savefig(save_gt_hm_path)
      # plt.show()

      # plt.imshow(batch_hm, cmap="gray")
      # plt.axis('off')  
      # plt.show()
      
      
      
      
      
      # input_h, input_w = self.opt.input_h, self.opt.input_w
      # # output_hw = opt.input_res // opt.down_ratio
      # output_h = opt.input_h // opt.down_ratio
      # output_w = opt.input_w // opt.down_ratio
      # over_hm = np.zeros((output_h, output_w, 3), dtype=np.float32)
      
      over_hm = np.zeros((batch_hm.shape[0],batch_hm.shape[1],3), dtype=np.float32)
      # print(opt.input_h)
      # print(over_hm.shape)
      # print(output_hm.shape)
      
      output_hm_norm = (output_hm-np.min(output_hm))/(np.max(output_hm)-np.min(output_hm))
      

      
      batch_hm_norm = (batch_hm-np.min(batch_hm))/(np.max(batch_hm)-np.min(batch_hm))
      
      over_hm[:,:,0] = np.squeeze(batch_hm_norm)
      over_hm[:,:,2] = np.squeeze(output_hm_norm)
      over_hm[:,:,1] = np.squeeze((batch_hm_norm*0.5  + output_hm_norm*0.5))
      
      
      # plt.imshow(over_hm)
      # plt.axis('off')  
      # plt.show()
      
      
      # over_hm = batch_hm + output_hm
      filename_over = f"{time_str}_over.jpg"
      
      save_over_path = os.path.join(image_folder,filename_over)
      if opt.heat_map_debug:
        plt.imsave(save_over_path, over_hm.squeeze(), cmap='gray')

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      # print(f"output['hm']: {output['hm'].shape}")
      # print(f"hm_loss: {hm_loss}")
      
      # print(f"output['reg']: {output['reg']}")
      # print(f"opt.reg_offset: {opt.reg_offset}")
      
      # print(f"opt.reg_offset: {opt.reg_offset}")

      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
      # print(f"batch['reg']: {batch['reg'].shape}")
      # print(f"batch['reg']: {batch['reg']}")
        
      # ------ pog_loss -------#
      
      # print(f"meta vp_s: {batch['meta']['vp_s']}")
      # if opt.pog_offset and opt.pog_weight > 0 :
      reg = output['reg'] if opt.reg_offset else None
      dets = ctdet_gaze_decode(output['hm'], reg=reg, K=opt.K)
      # print(f"dets.shape: {dets.shape}")
      dets = dets.detach().cpu().numpy()
      dets = dets.reshape(1, -1, dets.shape[2])
      # print(f"dets.shape: {dets.shape}")
      dets_out = ctdet_gaze_post_process(
            dets.copy(), batch['meta']['vp_c'].cpu().numpy(),
            batch['meta']['vp_s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
      cls_id = 1
      # dets_org_coord = dets_out[0][cls_id][0][:2]
      dets_org_coord = torch.tensor(dets_out[0][cls_id][:])
      dets_org_coord = dets_org_coord[:,:2] 
      # print(f"dets_out : {dets_out}")
      # print(f"dets_org_coord type: {type(dets_org_coord)}")
      # print(f"dets_org_coord.shape: {dets_org_coord.shape}")
      # print(f"dets_org_coord: {dets_org_coord}")

      
      dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
      dets_gt_out = ctdet_gaze_post_process(
            dets_gt.copy(), batch['meta']['vp_c'].cpu().numpy(),
            batch['meta']['vp_s'].cpu().numpy(),
            batch['hm'].shape[2], batch['hm'].shape[3], batch['hm'].shape[1])
      # dets_gt_org_coord = dets_gt_out[0][cls_id][0][:2]
      dets_gt_org_coord = torch.tensor(dets_gt_out[0][cls_id][:])
      dets_gt_org_coord = dets_gt_org_coord[:,:2]
      # print(f"dets_gt_org_coord: {dets_gt_org_coord}")
      # print(self.crit_pog(dets_org_coord, dets_gt_org_coord))
      pog_loss += self.crit_pog(dets_org_coord, dets_gt_org_coord) / opt.num_stacks
      # print(batch['hm'].shape)
      # print(f"pog_loss: {pog_loss}")
      # ------ pog_loss -------#
        
        
    # pog_weight = opt.pog_weight if opt.pog_offset_start_epoch > 10 else 0
    
    # loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss  
    loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss + opt.pog_weight * pog_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'off_loss': off_loss, 'pog_loss': pog_loss}
    return loss, loss_stats

class Ctdet_GazeTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(Ctdet_GazeTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'off_loss','pog_loss']
    loss = CtdetGazeLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_gaze_decode(
      output['hm'], reg=reg, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      vp_img = batch['vp'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)

      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      print("\n")
      # print(pred.shape)
      # print(vp_img.shape)
      # print(gt.shape)
      debugger.add_img(pred, 'pred_hm')
      debugger.add_img(gt, 'gt_hm')
      
      debugger.add_img(img, img_id='out_pred')
      
      print(len(dets[i]))
      for k in range(len(dets[i])):
        if dets[i, k, 2] > opt.center_thresh:
          debugger.add_gaze_point(dets[i, k, :2],dets[i, k, -1],dets[i, k, 2], img_id='out_pred')
          
      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 2] > opt.center_thresh:
          debugger.add_gaze_point(dets_gt[i, k, :2],dets_gt[i, k, -1],dets_gt[i, k, 2], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_gaze_decode(
      output['hm'], reg=reg, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_gaze_post_process(
      dets.copy(), batch['meta']['vp_c'].cpu().numpy(),
      batch['meta']['vp_s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]