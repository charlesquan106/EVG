#!/usr/bin/env python
import os
import pathlib
import math

import matplotlib.pyplot as plt

import numpy as np
import torch
import tqdm
from mpiifacegaze_eval_lib.utils.utils import AverageMeter
from mpiifacegaze_eval_lib.opts import opts
from mpiifacegaze_eval_lib.datasets.dataset_factory import get_dataset
from mpiifacegaze_eval_lib.models.model import create_model, load_model, save_model
from mpiifacegaze_eval_lib.models.decode import ctdet_gaze_decode
from mpiifacegaze_eval_lib.utils.post_process import ctdet_gaze_post_process


def euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def L2_distance(pred, gt):

    x1, y1 = pred
    x2, y2 = gt
    error = euclidean_distance(x1, y1, x2, y2)
    
    return error


def L2_distance_mm(pred, gt,mm_per_pixel):

    x1, y1 = pred
    x2, y2 = gt
    
    x1_mm = x1*mm_per_pixel
    y1_mm = y1*mm_per_pixel
    x2_mm = x2*mm_per_pixel
    y2_mm = y2*mm_per_pixel
    
    error = euclidean_distance(x1_mm, y1_mm, x2_mm, y2_mm)
    
    return error


def test(model, test_loader, opt):
    model.eval()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    
    model = model.to(device)
    
    L2_pixel_errors = AverageMeter()
    L2_mm_errors = AverageMeter()
    losses_gaze = AverageMeter()

    # predictions = []
    # gts = []
    with torch.no_grad():
        for iter_id, batch in enumerate(test_loader):
            print(f"Iteration {iter_id}/ {len(test_loader)}",end="\r")
            
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True) 
            outputs = model(batch['input'])
            # print(f"shape of input {batch['input'].shape}")
            
            # batch_image = batch['input']
            
            # output_image = batch_image[0].detach().cpu().numpy()
            # output_image = output_image.transpose(1, 2, 0)
            # plt.imshow(output_image, cmap="gray")
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()

            mm_per_pixel = batch['meta']['mm_per_pixel']
            mm_per_pixel = float(mm_per_pixel.detach().cpu().numpy())
            # print("mm_per_pixel",f'{mm_per_pixel}')
            
            img_id = batch['meta']['img_id']
            # print(f"img_id : {img_id}")
            # print(output)
            output = outputs[-1]
            hm = output['hm'].sigmoid_()
            
            # print(f"shape of hm {hm.shape}")
            
            # output_hm = output['hm'][0].detach().cpu().numpy()
            # output_hm = output_hm.transpose(1, 2, 0)
            
            # plt.imshow(output_hm, cmap="gray")
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            
            
            
            reg = output['reg'] if opt.reg_offset else None
            dets = ctdet_gaze_decode(hm, reg=reg, K=opt.K)
            # print(f"dets: {dets}")
            dets = dets.detach().cpu().numpy()
            dets = dets.reshape(1, -1, dets.shape[2])
            # dets[:, :, :2] *= opt.down_ratio
            # print(f"dets: {dets}")
            dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
            # print(f"dets_gt: {dets_gt}")
            # dets_gt[:, :, :2] *= opt.down_ratio
            # print(dets_gt)
            # print(dets_gt.shape)
            # for i in range(1):

            #     img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            #     vp_img = batch['vp'][i].detach().cpu().numpy().transpose(1, 2, 0)
            #     img = np.clip(((
            #         img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            # print(output.shape)
            dets_out = ctdet_gaze_post_process(
                dets.copy(), batch['meta']['vp_c'].cpu().numpy(),
                batch['meta']['vp_s'].cpu().numpy(),
                output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])

            # print(f"dets_out: {dets_out}")
            for i in range(1):
                cls_id = 1
                dets_org_coord = dets_out[i][cls_id][0][:2]
                # print(f"dets_org_coord: {dets_org_coord}")
            
            dets_gt_out = ctdet_gaze_post_process(
                dets_gt.copy(), batch['meta']['vp_c'].cpu().numpy(),
                batch['meta']['vp_s'].cpu().numpy(),
                batch['hm'].shape[2], batch['hm'].shape[3], batch['hm'].shape[1])

            # for i in range(1):
            #     cls_id = 1
            #     # print(f"dets_gt_out: {dets_gt_out[i][cls_id][0][:2]}")
            #     dets_gt_org_coord = dets_gt_out[i][cls_id][0][:2]
            #     print(f"dets_gt_org_coord: {dets_gt_org_coord}")
                
            dets_gt_org_coord = batch['meta']['vp_gazepoint']
            # print(f"vp_gazepoint : {dets_gt_org_coord}")
            
            #-------------- manual find heat map max ----------------
            output_hm = output['hm'][0].detach().cpu().numpy()
            # print(f"output_hm: {output_hm}")
            output_hm = output_hm.transpose(1, 2, 0)
            # max_index = torch.argmax(output_hm)
            # max_coord = torch.nonzero(max_index.view(-1) == torch.arange(output_hm.numel()), as_tuple=False)
            # max_coord_2d = (max_coord % output_hm.size(1), max_coord // output_hm.size(1))
            # print(f"output_hm max index: {max_coord_2d}")
               
            # plt.imshow(output_hm, cmap="gray")
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            
            batch_hm = batch['hm'][0].detach().cpu().numpy()
            # batch_hm = batch_hm.transpose(1,2,0)
            batch_hm = batch_hm.transpose(1, 2, 0)
            
            # plt.imshow(batch_hm, cmap="gray")
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            
            
            for i in range(1):
                cls_id = 1
                dets_coord = dets[0][0][:2]
                # print(f"dets_coord: {dets_coord}")
                
                dets_gt_coord = dets_gt[0][0][:2]
                # print(f"dets_gt_coord: {dets_gt_coord}")
                
            # output_hw = opt.input_res // opt.down_ratio
            # # print(f"output_hw: {output_hw}")
            # hm_over = np.zeros((output_hw, output_hw, 1), dtype=np.float32)
            # dets_coord_int = dets_coord.astype(np.int32)
            # dets_gt_coord_int = dets_gt_coord.astype(np.int32)
            
            # hm_over[dets_coord_int[1],dets_coord_int[0],0] =5  
            # hm_over[dets_gt_coord_int[1],dets_gt_coord_int[0],0] =15 
            # hm_over = output_hm + hm_over
            
            # output_hw = opt.input_res // opt.down_ratio
            # # print(f"output_hw: {output_hw}")
            # hm_over = np.zeros((output_hw, output_hw, 3), dtype=np.float32)
            hm_over = np.zeros((batch_hm.shape[0],batch_hm.shape[1],3), dtype=np.float32)
            
            dets_coord_int = dets_coord.astype(np.int32)
            dets_gt_coord_int = dets_gt_coord.astype(np.int32)

            # output_hm_min = np.min(output_hm)
            output_hm_norm = (output_hm-np.min(output_hm))/(np.max(output_hm)-np.min(output_hm))

            # for channel_i in range(3):
            #     hm_over[:,:,channel_i] = np.squeeze(output_hm_norm)
            # hm_over[dets_coord_int[1],dets_coord_int[0],0] =0  
            # hm_over[dets_coord_int[1],dets_coord_int[0],1] =0  
            # hm_over[dets_coord_int[1],dets_coord_int[0],2] =1  # blue
            # file_name = batch['meta']['file_name']
            # print(file_name)
            # hm_over[dets_gt_coord_int[1],dets_gt_coord_int[0],0] =1  # red
            # hm_over[dets_gt_coord_int[1],dets_gt_coord_int[0],1] =0  
            # hm_over[dets_gt_coord_int[1],dets_gt_coord_int[0],2] =0  

            
            
            
            L2_pixel_error = L2_distance(dets_org_coord, dets_gt_org_coord)
            L2_pixel_errors.update(L2_pixel_error)
            
            L2_mm_error = L2_distance_mm(dets_org_coord, dets_gt_org_coord,mm_per_pixel)
            L2_mm_errors.update(L2_mm_error)
            
            # L2_mm_error = L2_pixel_error*mm_per_pixel
            # L2_mm_errors.update(L2_mm_error)
            
            
            # print(f"L2_error = {L2_pixel_error} pixel, {L2_mm_error} mm")
            
            # plt.title(f"piexl error = {L2_pixel_error}")
            # plt.imshow(hm_over, cmap="gray")
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            # break

    mean_L2_pixel_errors = L2_pixel_errors.avg
    mean_L2_mm_errors = L2_mm_errors.avg

            
    return mean_L2_pixel_errors,mean_L2_mm_errors


def main(opt):
    print("mpiifacegaze eval")
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_res18_512_ep70/logs_2023-07-08-23-52/model_best.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep3_test/logs_2023-07-08-18-02/model_best.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70/logs_2023-07-08-20-11/model_best.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_256/logs_2023-07-09-17-49/model_45.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_one_person/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_all/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_res18_512_ep70_all/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_all_s_pos/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_all_keep_res/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_all_keep_res_resize_s_pos/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_all_keep_res_resize_s_pos_l_pog/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/cross_baseline/gaze_resdcn18_ep70_all_test_p04/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/csp/gaze_resdcn18_csp_p05/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/kr_resize/gaze_resdcn18_kr_resize_p05/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/csp_kr_resize/gaze_resdcn18_csp_kr_resize_p05/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/csp_kr_resize_pl/pl01/gaze_resdcn18_csp_kr_resize_pl01_p12/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_all_test_p14/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/csp_kr_resize_pl/pl001/gaze_resdcn18_ep70_all_keep_res_resize_s_pos_l_pog_0001/model_90.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/vp_s/gaze_resdcn18_test_vp_s_p05/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/sp_norm/gaze_resdcn18_ep70_sp_norm_p12/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/cross_baseline_sp_norm/gaze_resdcn18_ep70_all_base_sp_norm_p10/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/all_csp_kr_resize/gaze_resdcn18_ep70_all_csp_kr_resize_p10/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/all_csp_kr_resize_pl/pl001/gaze_resdcn18_ep70_all_csp_kr_resize_pl001_p10/model_50.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/cross_baseline_sp_norm_flipfix/gaze_resdcn18_ep70_all_base_sp_norm_flipfix_p02/model_70.pth"
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/cross_baseline_sp_norm_gp_shfit/gaze_resdcn18_ep70_all_base_sp_norm_gp_shift_p08/model_70.pth"
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep140_test/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep30_test_all/model_30.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep70_test_phone/model_70.pth"
    model_path = "/home/master_111/nm6114091/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep70_all/model_70.pth"
    
    
    model = load_model(model, model_path)
    # if opt.load_model != '':
    #     model, optimizer, start_epoch = load_model(
    #     model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    # Trainer = train_factory[opt.task]
    # trainer = Trainer(opt, model, optimizer)
    # trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    

    
    # test_loader = torch.utils.data.DataLoader(
    #   Dataset(opt, 'val'), 
    #   batch_size=1, 
    #   shuffle=False,
    #   num_workers=1,
    #   pin_memory=True
    # )
    # def should_exclude(idx,item):
    #     print(f"index: {idx}",end='\r')
    #     vp_gazepoint_x, vp_gazepoint_y = item['meta']['vp_gazepoint']

    #     return vp_gazepoint_x > opt.vp_w or vp_gazepoint_x < 0 or vp_gazepoint_y > opt.vp_h or vp_gazepoint_y < 0
  
    # exclude_val_index_list = [idx for idx, item in enumerate(Dataset(opt, 'val')) if should_exclude(idx,item)]
    # print("**********vp_gazepoint over virtual plane range need exclude**********")
    # print(f"exclude_val_index_list: {exclude_val_index_list}")
    # print(f"exclude_val_index_list len: {len(exclude_val_index_list)}")
    # exclude_val_sampler = torch.utils.data.sampler.SubsetRandomSampler([idx for idx in range(len(Dataset(opt, 'val'))) if idx not in exclude_val_index_list])

    
    
    test_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
    #   sampler=exclude_val_sampler,
      num_workers=1,
      pin_memory=True
    )
    
    

    
    # config = load_config()

    # output_rootdir = pathlib.Path(config.test.output_dir)
    # checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    # output_dir = output_rootdir / checkpoint_name
    # output_dir.mkdir(exist_ok=True, parents=True)
    # save_config(config, output_dir)

    # test_loader = create_dataloader(config, is_train=False)

    # model = create_model(config)
    # checkpoint = torch.load(config.test.checkpoint, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    

    L2_pixel_error,L2_mm_error = test(model, test_loader, opt)
    # print(L2_distance.shape)

    print(f'The mean error distance (pixel): {L2_pixel_error:.2f}')
    print(f'The mean error distance (mm): {L2_mm_error:.2f}')

    # output_path = output_dir / 'predictions.npy'
    # np.save(output_path, predictions.numpy())
    # output_path = output_dir / 'gts.npy'
    # np.save(output_path, gts.numpy())
    # output_path = output_dir / 'error.txt'
    # with open(output_path, 'w') as f:
    #     f.write(f'{angle_error}')


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
