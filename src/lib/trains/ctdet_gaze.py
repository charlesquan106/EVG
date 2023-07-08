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

class CtdetGazeLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetGazeLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, off_loss = 0, 0
    # print(f"outputs['hm']: {outputs['hm'].shape}")
    # print(f"batch['hm']: {batch['hm'].shape}")
    
    for s in range(opt.num_stacks):
      print(f"s: {s}")
      output = outputs[s]
      
      # output_hm = output['hm'][0].detach().cpu().numpy()
      # output_hm = output_hm.transpose(1, 2, 0)
      # print(f"output_hm_64,64: {output_hm[64][64]}")
      # print(f"output_hm_65,65: {output_hm[65][65]}")
      # plt.imshow(output_hm, cmap="gray")
      # plt.axis('off') 
      # plt.show()
      
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

      # output_hm_sig = _sigmoid(output['hm'][0])
      # print(f"output_hm_sig: {output_hm_sig.shape}")
      # output_hm_sig = output_hm_sig.permute(1, 2, 0)
      # output_hm_sig = output_hm_sig.detach().cpu().numpy()
      # print(f"output_hm_sig_64,64: {output_hm_sig[64][64]}")
      # print(f"output_hm_sig_65,65: {output_hm_sig[65][65]}")
      
      # batch_hm = batch["hm"][0].detach().cpu().numpy()
      # print(f"batch_hm: {batch_hm.shape}")
      # # batch_hm = batch_hm.transpose(1,2,0)
      # batch_hm = batch_hm.transpose(1, 2, 0)

      # # plt.imshow(output_hm_sig, cmap="gray")
      # # plt.axis('off')  
      # # plt.show()
      
      # plt.imshow(batch_hm, cmap="gray")
      # plt.axis('off')  
      # plt.show()

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      # print(f"hm_loss: {hm_loss}")
      
      # print(f"output['reg']: {output['reg']}")
      # print(f"opt.reg_offset: {opt.reg_offset}")

      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
        
    loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'off_loss': off_loss}
    return loss, loss_stats

class Ctdet_GazeTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(Ctdet_GazeTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'off_loss']
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