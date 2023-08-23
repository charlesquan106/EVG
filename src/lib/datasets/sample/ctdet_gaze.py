from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math


class CTDet_gazeDataset(data.Dataset):
  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    # print(f"img_id{img_id}")
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    # print(num_objs)
  
    img = cv2.imread(img_path)
    if self.opt.resize_raw_image:
      img = cv2.resize(img, (self.opt.resize_raw_image_w, self.opt.resize_raw_image_h), interpolation=cv2.INTER_LINEAR)
    img_height, img_width = img.shape[0], img.shape[1]
    input_h, input_w = self.opt.input_h, self.opt.input_w
    # print(img_height)
    # print(img_width)
    # print(input_h)
    # print(input_w)

    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (img_height | self.opt.pad) + 1
      input_w = (img_width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img_width, img_height) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    # print(self.opt.keep_res)
    # print(self.opt.pad)
    # print(f"input_w, input_h: {input_w}, {input_h}")

   
    
    flipped = False
    if self.split == 'train':
      # pass
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :].copy()
      # flipped = True
      # img = img[:, ::-1, :].copy()
        
    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)

    inp = (inp.astype(np.float32) / 255.)
    # if self.split == 'train' and not self.opt.no_color_aug:
    #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    # inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    
    # print(f"output_hw: {output_h},{output_w}")

    num_classes = self.num_classes
  
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    face_grid = np.zeros((self.max_objs, 2), dtype=np.int64)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
                    
      
      
    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      
      unit_mm,unit_pixel = ann["screenSize"]
      unit_mm = eval(unit_mm)
      raw_sc_width_mm, raw_sc_height_mm = unit_mm  
      
      unit_pixel = eval(unit_pixel)
      raw_sc_width, raw_sc_height = unit_pixel  
      # print("screenSize unit_pixel",f'{sc_height}, {sc_width}')
      # sc = np.ones((1, sc_width, sc_height), dtype=np.float32)
      
      # print("file_name",f'{file_name}')
      # print("unit_mm",f'{sc_width_mm} {sc_height_mm}')
      
      # 5 pixel/mm  norm_screen_plane
      if self.opt.vp_pixel_per_mm > 0 :
        sc_height = int(raw_sc_height_mm * self.opt.vp_pixel_per_mm)
        sc_width = int(raw_sc_width_mm * self.opt.vp_pixel_per_mm)
      else:
        sc_height = raw_sc_height
        sc_width = raw_sc_width
     
      vp_width, vp_height = self.opt.vp_w, self.opt.vp_h
      # vp = np.zeros((1, vp_width, vp_height), dtype=np.float32)
      vp = np.zeros((vp_height, vp_width , 1), dtype=np.float32)
      vp = vp.transpose(1,2,0)
      
      
      
      
      raw_x,raw_y = ann['gazepoint']
      
      # Random Gazepoint
      # 1 cm (mm *10) as Gazepoint shift distance
      if self.split == 'train':
          if not self.opt.no_shift_gaze_point_aug : 
            raw_x = np.random.uniform()* self.opt.vp_pixel_per_mm*10 + raw_x
            raw_y = np.random.uniform()* self.opt.vp_pixel_per_mm*10 + raw_y

      
      # print(self.opt.vp_pixel_per_mm)
      if self.opt.vp_pixel_per_mm > 0 :
        x = int((raw_x / raw_sc_width) * sc_width)
        y = int((raw_y / raw_sc_height) * sc_height)
      else:
        x = raw_x
        y = raw_y
      # print("raw gazepoint",f'{x} {y}')
      # print("raw sc_size ",f'{raw_sc_width} {raw_sc_height}')
      
      # print("gazepoint",f'{x} {y}')
      # print("sc_size ",f'{sc_width} {sc_height}')
      if flipped:
        x = sc_width - x - 1
      
      ann_id = ann['id']
      # print(f"id: {ann_id}")
      sc_gazepoint = np.array([x,y],dtype=np.int64)

      if self.opt.camera_screen_pos:
        camera_screen_offset = sc_height/2
      else:
        camera_screen_offset = 0
        # print(f"sc_gazepoint: {sc_gazepoint}")
      # vp_gazepoint = [(vp_width-sc_width)/2+sc_gazepoint[0] ,(vp_height-sc_height)/2+sc_gazepoint[1]+camera_screen_offset]
      vp_gazepoint = [(vp_width/2)+(sc_gazepoint[0]-(sc_width/2)) ,(vp_height/2)+(sc_gazepoint[1]-(sc_height/2))+camera_screen_offset]
      # print(f"vp_gazepoint: {vp_gazepoint}")
      # **************


      if flipped:
        vp = vp[:, ::-1, :].copy()
          
      vp_c = np.array([ vp_width / 2.0, vp_height/ 2.0], dtype=np.float32)
      if self.opt.keep_res:
        # trans_output_h = (vp_height | self.opt.pad) + 1
        # trans_output_w = (vp_width | self.opt.pad) + 1
        # vp_s = np.array([trans_output_w, trans_output_h], dtype=np.float32)
        trans_output_h = (vp_height | self.opt.pad) + 1
        trans_output_w = (vp_width | self.opt.pad) + 1
        vp_s = np.array([vp_width, vp_height], dtype=np.float32)
      else:
        vp_s = max(vp_width, vp_height) * 1.0
    #   print(f"vp_s: {vp_s}")
      
      # print(f"output_w, output_h: {output_w}, {output_h}")
          
      trans_vp2out = get_affine_transform(vp_c, vp_s, 0, [output_w, output_h])
      # vp_trans_out = cv2.warpAffine(vp, trans_vp2out, 
      #                   (output_w, output_h),
      #                   flags=cv2.INTER_LINEAR)

      # print(f"vp_out: {vp_trans_out.shape[0]},{vp_trans_out.shape[1]}")
      
      
      cls_id = 0
    #   print(vp_gazepoint)
      vp_gazepoint_output = affine_transform(vp_gazepoint, trans_vp2out)
      # print(vp_gazepoint_output)
      h, w = self.opt.vp_heatmap_hw, self.opt.vp_heatmap_hw
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array([ vp_gazepoint_output[0], vp_gazepoint_output[1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        
        # hm_gazepoint_idx = np.where(hm[cls_id] == 20)
        # print(f"hm_gazepoint_idx: {hm_gazepoint_idx}")
        
        # avoid exceess the max of ind which accroding to output_h*output_w 
        if ct_int[1] >= output_h :
          ind[k] = (output_h-1) * output_w + ct_int[0]
        else:
          ind[k] = ct_int[1] * output_w + ct_int[0]
        # ind[k] = np.clip(ind[k],0,output_w*output_h)
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        face_grid[k] = 1
        # print('---------------')
        # print("ct",f'{ct}')
        # print("ct_int",f'{ct_int}')
        # print("reg[k]",f'{reg[k]}')
        # print("radius",f'{radius}')

        gt_det.append([ct[0], ct[1], 1, cls_id])

        
    # print("vp_trans_out.shape",f'{vp_trans_out.shape}') 
    # print("ct_int",f'{ct_int}') 
    # vp_trans_out[ct_int[1],ct_int[0]] = 1
    # hm  = hm + vp_trans_out
    
    # ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'vp' : vp}
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind}
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.face_grid:
      ret.update({'face_grid': face_grid})
    # if self.opt.debug > 0 or not self.split == 'train':
    pog_loss = 1
    if pog_loss:  
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 4), dtype=np.float32)
      meta = {'c': c, 's': s,'vp_c': vp_c, 'vp_s': vp_s, 'gt_det': gt_det, 'img_id': img_id,'vp_gazepoint': vp_gazepoint}
      ret['meta'] = meta
    return ret