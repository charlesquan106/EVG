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
from mpiifacegaze_eval_lib.models.utils import _sigmoid
from mpiifacegaze_eval_lib.utils.common import angular_error,calculate_combined_gaze_direction_no_h, calculate_combined_gaze_direction, calculate_combined_gaze_direction_normalize
import cv2


# def euclidean_distance(x1, y1, x2, y2):
#     distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#     return distance

# def L2_distance(pred, gt):

#     x1, y1 = pred
#     x2, y2 = gt
#     error = euclidean_distance(x1, y1, x2, y2)
    
#     return error

# def L2_distance_mm(pred, gt,mm_per_pixel):

#     x1, y1 = pred
#     x2, y2 = gt
    
#     x1_mm = x1*mm_per_pixel
#     y1_mm = y1*mm_per_pixel
#     x2_mm = x2*mm_per_pixel
#     y2_mm = y2*mm_per_pixel
    
#     error = euclidean_distance(x1_mm, y1_mm, x2_mm, y2_mm)
    
#     return error




class UpdateHeatmap(object):
    def __init__(self,w,h):
        self.h = h
        self.w = w
        self.heatmap = np.zeros((self.w,self.h))
        self.sum = np.zeros((self.w,self.h))
        self.counter = np.zeros((self.w,self.h))

    def update(self, x, y , val):
        wide = 20
        # self.sum[x-wide: x+wide, y-wide: y+wide , 0] = int(np.log(val + 1))
        self.sum[x-wide: x+wide, y-wide: y+wide ] = int(val)
    
        self.sum[x-wide: x+wide, y-wide: y+wide] = self.sum[x-wide: x+wide, y-wide: y+wide] + self.heatmap[x-wide: x+wide, y-wide: y+wide]
        self.counter[x-wide: x+wide, y-wide: y+wide] = self.counter[x-wide: x+wide, y-wide: y+wide] + 1
        if self.counter[x, y] > 1 :
            self.heatmap[x-wide: x+wide, y-wide: y+wide] = self.sum[x-wide: x+wide, y-wide: y+wide] / self.counter[x-wide: x+wide, y-wide: y+wide]
        else:
            self.heatmap[x-wide: x+wide, y-wide: y+wide] = self.sum[x-wide: x+wide, y-wide: y+wide]
        
    def quantized(self,x_scale,y_scale):
        
        quantized_data = np.zeros((x_scale, y_scale))
        
        x_unit = int(self.w / x_scale)
        y_unit = int(self.h / y_scale)
        print(x_unit)
        
        for i in range(x_scale):
            for j in range(y_scale):
                quantized_data[i, j] = np.mean(self.heatmap[i * x_unit:(i + 1) * x_unit, j * y_unit:(j + 1) * y_unit])
        

        return quantized_data


def unit_error_display(heatmap, dets_gt_org_coord, error):


    x,y = dets_gt_org_coord
    heatmap[x, y] = error
    
    pass


def test(model, test_loader, opt):
    
    # heatmap = np.zeros((opt.vp_h,opt.vp_w))
    
    
    model.eval()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    
    model = model.to(device)
    
    L2_pixel_errors = AverageMeter()
    L2_mm_errors = AverageMeter()
    x_pixel_errors = AverageMeter()
    y_pixel_errors = AverageMeter()
    x_mm_errors = AverageMeter()
    y_mm_errors = AverageMeter()
    
    angle_errors = AverageMeter()
    yaw_errors = AverageMeter()
    pitch_errors = AverageMeter()
    

    update_heatmap = UpdateHeatmap(opt.vp_w,opt.vp_h)
    
    L2_pixel_errors_list = []

    # predictions = []
    # gts = []
    with torch.no_grad():
        for iter_id, batch in enumerate(test_loader):
            print(f"Iteration {iter_id}/ {len(test_loader)}",end="\r")
            
            # if iter_id > 100:
            #     break
            
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True) 
            outputs = model(batch['input'])
            
            output = outputs[-1]
            hm = output['hm'].sigmoid_()
            # print(f"shape of input {batch['input'].shape}")
            
            batch_image = batch['input']
            
            batch_image = batch_image[0].detach().cpu().numpy()
            batch_image = batch_image.transpose(1, 2, 0)
            
            gaze_origin_tensor = batch['meta']['gaze_origin_tensor']
            camera_transformation_tensor = batch['meta']['camera_transformation_tensor']
            head_rvec_tensor = batch['meta']['head_rvec_tensor']
            gaze_R_tensor = batch['meta']['gaze_R_tensor']
            # print("gaze_origin_tensor : ", gaze_origin_tensor)
            # print("camera_transformation_tensor : ", camera_transformation_tensor)
            # print("head_rvec_tensor : ", head_rvec_tensor)
            

            # print(batch_image.shape)
            
            # plt.figure(figsize=(12, 4))
            # plt.subplot(1, 3, 1)
            # plt.imshow(batch_image)
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            if opt.face_hm_head: 

                batch_face_hm = batch["face_hm"]
                batch_face_hm = batch_face_hm[0].detach().cpu().numpy()
                batch_face_hm = batch_face_hm.transpose(1,2,0)
                # plt.subplot(1, 3, 2)
                # plt.imshow(batch_face_hm)
                # plt.axis('off')  # 关闭坐标轴
                # plt.show()
                
                
                output_face_hm = _sigmoid(output['face_hm'][0]).detach().cpu().numpy()
                output_face_hm = output_face_hm.transpose(1, 2, 0)
                # plt.subplot(1, 3, 3)
                # plt.imshow(output_face_hm)
                # plt.show()
                
            
            
            # plt.figure(figsize=(16, 4))
            # plt.subplot(1, 3, 1)
            # plt.imshow(batch_image, cmap="gray")
            # plt.title('input image')
            
            mm_per_pixel = torch.tensor(batch['meta']['mm_per_pixel'].numpy(), dtype=torch.float32)
            
            # print(type(mm_per_pixel))
            # print("mm_per_pixel = ", mm_per_pixel)
            # mm_per_pixel = torch.tensor(batch['meta']['mm_per_pixel'].numpy())
            # mm_per_pixel = mm_per_pixel.view(N, 1).expand(N, 2)
            # mm_per_pixel = float(mm_per_pixel.detach().cpu().numpy())
            # print("mm_per_pixel",f'{mm_per_pixel}')
            
            img_id = batch['meta']['img_id']
            # print(f"img_id : {img_id}")
            # print(output)
            
            
            
            # heatmap Ground Truth vs Predict
            gt_hm = batch['hm'][0].detach().cpu().numpy()
            gt_hm = gt_hm.transpose(1, 2, 0)
    
            output_hm = output['hm'][0].detach().cpu().numpy()
            output_hm = output_hm.transpose(1, 2, 0)
            
            # plt.subplot(1, 3, 2)
            # plt.imshow(gt_hm, cmap="gray")
            # plt.title('Ground Truth')

            # plt.subplot(1, 3, 3)
            # plt.imshow(output_hm, cmap="gray")
            # plt.title('Predict')
            
            
            
            
            reg = output['reg'] if opt.reg_offset else None
            # reg = None
            dets = ctdet_gaze_decode(hm, reg=reg, K=opt.K)
            # print(f"dets: {dets}")
            dets = dets.detach().cpu().numpy()
            dets = dets.reshape(1, -1, dets.shape[2])
            # print(dets)

            dets_out = ctdet_gaze_post_process(
                dets.copy(), batch['meta']['vp_c'].cpu().numpy(),
                batch['meta']['vp_s'].cpu().numpy(),
                output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
            
            # print(dets_out)

            
            cls_id = 1
            dets_org_coord = torch.tensor(dets_out[0][cls_id][:])
            dets_org_coord = dets_org_coord[:,:2] 
            # print(f"dets_org_coord type: {type(dets_org_coord)}")
            # print(f"dets_org_coord.shape: {dets_org_coord.shape}")
            # print(f"dets_org_coord: {dets_org_coord}")
            # for i in range(1):
            #     cls_id = 1
            #     dets_org_coord = torch.tensor(dets_out[0][cls_id][:])
            #     dets_org_coord = dets_out[i][cls_id][0][:2]
            #     print(f"dets_org_coord type: {type(dets_org_coord)}")
            #     print(f"dets_org_coord.shape: {dets_org_coord.shape}")
            #     # print(f"dets_org_coord: {dets_org_coord}")
            
                
            dets_gt_org_coord = torch.tensor(batch['meta']['vp_gazepoint'].numpy())
            # print("dets_gt_org_coord: ",dets_gt_org_coord )
            # vp_gazepoint (w, h)
            # print(f"vp_gazepoint type : {type(dets_gt_org_coord)}")
            # print(f"vp_gazepoint : {dets_gt_org_coord}")
            
            
            
            
            error = 0
            error = torch.sum((dets_org_coord - dets_gt_org_coord)**2, dim=1)
            L2_pixel_error = torch.sqrt(error)
            L2_pixel_error = L2_pixel_error.mean()
            L2_pixel_errors.update(L2_pixel_error)
            
            # print(dets_org_coord)
            # print(dets_org_coord[0][0])
            # print(dets_gt_org_coord[0][0])
            
            # print(dets_org_coord[0][1])
            # print(dets_gt_org_coord[0][1])
            # print(dets_org_coord.shape)
            
            error_x = abs(torch.sum((dets_org_coord[0][0] - dets_gt_org_coord[0][0])).mean())
            error_y = abs(torch.sum((dets_org_coord[0][1] - dets_gt_org_coord[0][1])).mean())
            # print(error)
            # print(L2_pixel_error)
            # print(error_x,"   ", error_y)
            # print(mm_per_pixel)
            
            x_pixel_errors.update(error_x)
            y_pixel_errors.update(error_y)
            x_mm_error = error_x*(mm_per_pixel[0][0])
            x_mm_error = x_mm_error.mean()
            x_mm_errors.update(x_mm_error)
            y_mm_error = error_y*(mm_per_pixel[0][1])
            y_mm_error = y_mm_error.mean()
            y_mm_errors.update(y_mm_error)
            
            
            L2_mm_error = torch.sqrt(error*(mm_per_pixel[0]**2))
            L2_mm_error = L2_mm_error.mean()
            L2_mm_errors.update(L2_mm_error)
            
            
            # unit_error_display(dets_gt_org_coord,error)
            
            x = int(dets_gt_org_coord[0][0])
            y = int(dets_gt_org_coord[0][1])
            # print("dets_gt_org_coord: ", x,"  ", y )
            # print(head_rvec_tensor)
            
            # head_rotation = [cv2.Rodrigues(rvec) for rvec in head_rvec_tensor]
            head_rotation_np = head_rvec_tensor.numpy()
            # print("head_rotation_np = ",head_rotation_np[0])
            # print("head_rotation_np.shape = ", head_rotation_np[0].shape)
            head_rotation,_ = cv2.Rodrigues(head_rotation_np[0])
            # print("head_rotation = ",head_rotation)
            # print(type(head_rotation))
            head_rotation_tensor = torch.tensor(head_rotation)
            
            dets_gt_sc_org_coord = torch.tensor(batch['meta']['sc_gazepoint'].numpy(), dtype=torch.float32)
            # print("dets_gt_sc_org_coord = ", dets_gt_sc_org_coord)
            
            # dets_sc_org_coord = torch.tensor(dets_org_coord[0]
            dets_sc_org_coord = torch.tensor([[(dets_org_coord[0][0]-960),(dets_org_coord[0][1]-1180)]], dtype=torch.float32)
            
            # width  vp + (sc/2) - (vp/2) =  vp + (1920/2) - (3840/2) = vp -  960
            # height  vp + (sc/2) - (vp/2) - camera_screen_y_offset(sc_height/2) =  vp + (1080/2) - (2360/2) - (1080/2)  =  vp - 1180
            

            
            # print("dets_sc_org_coord = ", dets_sc_org_coord)
            
            # vp_gazepoint = np.array([(vp_width/2)+(sc_gazepoint[0]-(sc_width/2))+ camera_screen_x_offset, (vp_height/2)+(sc_gazepoint[1]-(sc_height/2))+camera_screen_y_offset], dtype=np.float32)
 
            dets_gt_sc_org_coord_mm = torch.tensor([[dets_gt_sc_org_coord[0][0]*mm_per_pixel[0][0],dets_gt_sc_org_coord[0][1]*mm_per_pixel[0][1]]], dtype=torch.float32)
            dets_sc_org_coord_mm = torch.tensor([[dets_sc_org_coord[0][0]*mm_per_pixel[0][0],dets_sc_org_coord[0][1]*mm_per_pixel[0][1]]], dtype=torch.float32)           
            
            # dets_gt_org_coord_mm = torch.tensor([[dets_gt_org_coord[0][0]*mm_per_pixel[0][0],dets_gt_org_coord[0][1]*mm_per_pixel[0][1]]], dtype=torch.float32)
            # dets_org_coord_mm = torch.tensor([[dets_org_coord[0][0]*mm_per_pixel[0][0],dets_org_coord[0][1]*mm_per_pixel[0][1]]], dtype=torch.float32)
            # print("dets_gt_org_coord_mm = ", dets_gt_org_coord_mm )
            # print("dets_org_coord_mm = ", dets_org_coord_mm )
            
            
            direction_gt = calculate_combined_gaze_direction_normalize(gaze_origin_tensor, dets_gt_sc_org_coord_mm,camera_transformation_tensor, gaze_R_tensor)           
            # direction_gt = calculate_combined_gaze_direction_normalize(gaze_origin_tensor, dets_gt_org_coord_mm,camera_transformation_tensor, gaze_R_tensor)
            # direction_gt = calculate_combined_gaze_direction_no_h(gaze_origin_tensor, dets_gt_org_coord,camera_transformation_tensor)
            # print("direction_gt: ",direction_gt)
            
            direction = calculate_combined_gaze_direction_normalize(gaze_origin_tensor, dets_sc_org_coord_mm,camera_transformation_tensor, gaze_R_tensor)
            # direction = calculate_combined_gaze_direction_normalize(gaze_origin_tensor, dets_org_coord_mm,camera_transformation_tensor, gaze_R_tensor)
            # direction = calculate_combined_gaze_direction_no_h(gaze_origin_tensor, dets_org_coord,camera_transformation_tensor)

            
            # cal_angular_error = angular_error(direction_gt,direction).mean()
            # print("angle error :",cal_angular_error)
            
            # print(direction_gt)
            # print(direction)
            
            
            direction_error = abs(np.degrees(direction_gt[0])- np.degrees(direction[0]))
            # print("direction_error = ", direction_error)
            yaw_errors.update(direction_error[0])
            pitch_errors.update(direction_error[1])
            # print(direction_error)
            angle_error = angular_error(direction_gt,direction)
            # print("angle_error = ", angle_error)
            # print("angle_error type = ",type(angle_error))
            angle_error = angle_error.mean()
            # print("angle_error type  mean = ",type(angle_error))
            angle_errors.update(abs(angle_error))
            # angle_error = angular_error(direction_a,direction_b)
            # angle_errors.update(angle_error)
            # angle_errors.update(angular_error(direction_gt,direction).mean())
            
            
            update_heatmap.update(x,y,L2_pixel_error)
            
            
            
            
            
            
            # heatmap = update_heatmap.vis()
            # plt.title(f"piexl error heatmap")
            # plt.imshow(heatmap, cmap="gray")
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()      
        
            
            # print(f"L2_error = {L2_pixel_error} pixel, {L2_mm_error} mm")
            
            # plt.title(f"piexl error = {L2_pixel_error}")
            # plt.imshow(hm_over, cmap="gray")
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            # break
            
            # 添加说明文字
            # plt.figtext(0.5, 0.01, f"L2_error = {L2_pixel_error} pixel", ha='center', fontsize=12)
            # plt.show()
            
            
            L2_pixel_errors_list.append(int(L2_pixel_error))
    
    x_scale, y_scale = 50,50
    heatmap = update_heatmap.quantized(x_scale,y_scale)
    heatmap = heatmap.transpose(1,0)
    
    x_unit = opt.vp_w / x_scale
    y_unit = opt.vp_h / y_scale
    extents=[0, x_scale, y_scale, 0]
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap=plt.cm.jet, extent= extents)
    # 添加 x 和 y 轴标签
    plt.xlabel(f'X ({x_unit} pixel/unit)')
    plt.ylabel(f'Y ({y_unit} pixel/unit)')

    # 在每10个 bin 上添加数值标注
    step = 5
    x_labels = np.arange(0, x_scale, step)
    y_labels = np.arange(0, y_scale, step)
    
    
    plt.xticks(x_labels, [str(i) for i in x_labels])
    plt.yticks(y_labels, [str(i) for i in y_labels])
    
    
    # x_labels = np.linspace(0, 50, 11)  # 使用np.linspace设置刻度位置
    # y_labels = np.linspace(0, 50, 11)

    # # 设置 x 和 y 轴的刻度标签
    # plt.xticks(x_labels, [str(int(i)) for i in x_labels])
    # plt.yticks(y_labels, [str(int(i)) for i in y_labels])
    


    # 创建颜色条
    cbar = plt.colorbar()
    cbar.set_label('Pixel Error')
    
    # plt.grid(color='black', linestyle='-', linewidth=0.5,)
    
    for i in range(1, x_scale):
        plt.axvline(x=i, color='black', linewidth=0.5)
    for j in range(1, y_scale):
        plt.axhline(y=j, color='black', linewidth=0.5)

    plt.title('Quantized Data Visualization')
    # plt.show()
    
    
    

    mean_L2_pixel_errors = L2_pixel_errors.avg
    mean_x_pixel_errors = x_pixel_errors.avg
    mean_y_pixel_errors = y_pixel_errors.avg
    mean_x_mm_errors = x_mm_errors.avg
    mean_y_mm_errors = y_mm_errors.avg
    mean_L2_mm_errors = L2_mm_errors.avg
    
    mean_yaw_errors = yaw_errors.avg
    mean_pitch_errors = pitch_errors.avg
    
    print("angle_errors = ", angle_errors.sum)
    mean_angle_errors = angle_errors.avg
    
    # Creating histogram
    # n_bins = 20
    # fig, axs = plt.subplots(1, 1,
    #                         figsize =(10, 7),
    #                         tight_layout = True)
    # axs.hist(L2_pixel_errors_list, bins = n_bins)
 
    # # Show plot
    # plt.show()
    
    n_bins = 20
    fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    hist, bins, _ = axs.hist(L2_pixel_errors_list, bins=n_bins, alpha=0.5, color='blue')
    
    # 計算每個 bin 中的百分比
    bin_percentages = ((hist/10)/ len(L2_pixel_errors_list)) * 100

    # 添加 y 軸標籤為百分比
    axs.set_ylabel('Percentage (%)')

    # 添加 x 軸標籤和標題
    axs.set_xlabel('L2 Pixel Errors')
    axs.set_title('Histogram of L2 Pixel Errors')
    
    
    # plt.yticks(np.arange(0, max(max(hist), max(hist_normal)) + 1, 1))  # 根據兩組數據的最大值設定刻度

    # # 添加圖例
    # axs.legend()

    # 在每個 bin 上標註百分比
    for i in range(len(bins) - 1):
        bin_center = (bins[i] + bins[i + 1]) / 2
        plt.text(bin_center, hist[i] + 0.2, f'{bin_percentages[i]*10:.1f}%', ha='center', va='bottom')
        
        
        
        # 設定 y 軸刻度，自動調整以符合資料的比例
    # plt.yticks(np.arange(0, max(hist) + 1))  # 設定刻度，此處以 1 為間隔

    # 顯示圖形
    # plt.show()
    
    
    print(f'The mean error distance (pixel) / (mm): {mean_L2_pixel_errors:.2f} / {mean_L2_mm_errors:.2f}')
    print(f'The mean error x (pixel) / (mm): {mean_x_pixel_errors:.2f} / {mean_x_mm_errors:.2f} ')
    print(f'The mean error y (pixel) / (mm): {mean_y_pixel_errors:.2f} / {mean_y_mm_errors:.2f}')
    # print(f'The mean error distance (mm): {mean_L2_mm_errors:.2f}')
    
    print(f'The mean yaw error (degree): {mean_yaw_errors:.5f}')
    print(f'The mean pitch error (degree): {mean_pitch_errors:.5f}')
    print(f'The mean angle error (degree): {mean_angle_errors:.5f}')



            
    # return mean_L2_pixel_errors, mean_L2_mm_errors, mean_x_pixel_errors, mean_y_pixel_errors


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
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/all_csp_kr_resize_pl/pl001/gaze_resdcn18_ep70_all_csp_kr_resize_pl001_p10/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/cross_baseline_sp_norm_flipfix/gaze_resdcn18_ep70_all_base_sp_norm_flipfix_p02/model_70.pth"
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/cross_baseline_sp_norm_gp_shfit/gaze_resdcn18_ep70_all_base_sp_norm_gp_shift_p08/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_all_norm_csp_kr_pl001_p05_pl_fix/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_csp_kr_resize_p05_petrain_eve/model_56.pth"
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep140_test/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep30_test_all/model_30.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep70_all_phone_pl01/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep70_all/model_70.pth"
    
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcncut_18_mpii_r10_ep70_all_norm_csp_kr_pl001_p05/model_14.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcncut_18_mpii_r10_ep70_all_norm_csp_kr_pl001_p05/model_14.pth" 
    
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_eve_sc_kr/model_5.pth"
    
    #### eve ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_pl001_2/model_2.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/mobv2/gaze_eve_mobv2_40_64_480/model_2.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/mobv2/gaze_eve_mobv2_40_64_640/model_1.pth"
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/mobv2035/gaze_eve_mobv2_035_40_64_640_no_pre/model_2.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/mobv2035/gaze_eve_mobv2_035_40_64_480_no_pre/model_4.pth"
    
    
    
    
    #### eve - face ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/eve/resdcnface_18/gaze_eve_resdcnface_18_eve_ep20/model_2.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/eve/resdcnface_18/gaze_eve_resdcnface_18_eve_ep10_f20/model_1.pth"
    
    #### gazecapture ####    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/gazecapture/resdcn_18/gaze_gazecapture_ep70_all_hm_adapt_r_no_csp/model_15.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/gazecapture/resdcn_18/gaze_gazecapture_all_no_scp_pl01/model_58.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/gazecapture/resdcn_18/gaze_gazecapture_all_no_scp_pl001/model_53.pth"
    
    #### gazecapture - face ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/gazecapture/resdcnface_18/gaze_gc_resdcnface_18_all_f2/model_14.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/gazecapture/resdcnface_18/gaze_gc_resdcnface_18_all_f025/model_18.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/gazecapture/resdcnface_18/gaze_gazecapture_all_no_scp_f001/model_35.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/gazecapture/resdcnface_18/gaze_gazecapture_all_no_scp_f01/model_34.pth"
    
    
    
    #### cross mpii vs himax ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_cross_mpii_himax_gray/model_1.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_impii_resdcn18_p05_test_himax_sc_kr_mono/model_44.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/cross_mpii_himax/resdcn_18/gaze_cross_mpii_himax_laptop_gray_vp_large/model_12.pth"
    
    
    #### cross eve vs himax ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_eve_test_himax_sc_kr_mono/model_2.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/cross_eve_himax/gaze_cross_eve_himax_gray/model_1.pth"
    
    #### cross pretrain eve  himax ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_eve_weight_himax_gray_lr125_4_decay/model_10.pth"
    
    #### cross pretrain mpii  himax ####   
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_mpii_weight_himax_gray/model_2.pth"
    
    
    #### cross pretrain eve  himax(train) ####
    # mono test
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_eve_weight_himax_sp_Ben_all_gray_lr125_4/model_8.pth"
    # rgb test
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_eve_weight_himax_all_rgbtest_gray_lr125_4/model_10.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/mobv2035/gaze_eve_mobv2035_40_64_480_weight_himax_all_rgb/model_13.pth"
    model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/mobv2/gaze_eve_mobv2_40_64_480_weight_himax_all_rgb/model_13.pth"
    
    
    #### cross pretrain mpii  himax(train) ####  
    # mono test 
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_mpii_weight_himax_sp_Ben_all_gray_lr125_4/model_8.pth"
    # rgb test
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_mpii_weight_himax_all_rgbtest_gray_lr125_4/model_7.pth"
    
    #### cross pretrain eve himax ct17_cy30   ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_eve_weight_himax_all_ct17_cy30_gray_lr125_4/model_11.pth"
    
    #### cross pretrain mpii himax ct17_cy30   ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_mpii_weight_himax_all_ct17_cy30_gray_lr125_4/model_1.pth"
    
    
    #### cross pretrain eve himax test spilt   ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_eve_weight_himax_all_test_spilt_gray_lr125_4/model_36.pth"
    
    #### cross pretrain mpii himax test spilt   ####
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/himax/resdcn_18/gaze_mpii_weight_himax_all_test_spilt_gray_lr125_4/model_11.pth"
    
    
    model = load_model(model, model_path)
    # if opt.load_model != '':
    #     model, optimizer, start_epoch = load_model(
    #     model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    # Trainer = train_factory[opt.task]
    # trainer = Trainer(opt, model, optimizer)
    # trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    

    
    test_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
    )
    
    
    # dataset_exclude = Dataset(opt, 'val')
    
    # list_dataset_exclude = dataset_exclude.show_filter_data
    # print("list_dataset_exclude:",list_dataset_exclude)
    
    # def should_exclude(idx,item):
    #     print(f"index: {idx}",end='\r')
    #     vp_gazepoint_x, vp_gazepoint_y = item['meta']['vp_gazepoint']

    #     return vp_gazepoint_x > opt.vp_w or vp_gazepoint_x < 0 or vp_gazepoint_y > opt.vp_h or vp_gazepoint_y < 0
  
    # exclude_val_index_list = [idx for idx, item in enumerate(Dataset(opt, 'val')) if should_exclude(idx,item)]
    # print("**********vp_gazepoint over virtual plane range need exclude**********")
    # print(f"exclude_val_index_list: {exclude_val_index_list}")
    # print(f"exclude_val_index_list len: {len(exclude_val_index_list)}")
    # exclude_val_sampler = torch.utils.data.sampler.SubsetRandomSampler([idx for idx in range(len(Dataset(opt, 'val'))) if idx not in exclude_val_index_list])


    # def filter_data(item):
    #     # 根据需要定义筛选条件
        
    #     vp_gazepoint_x, vp_gazepoint_y = item['meta']['vp_gazepoint']
        
    #     if vp_gazepoint_x > opt.vp_w or vp_gazepoint_x < 0 or vp_gazepoint_y > opt.vp_h or vp_gazepoint_y < 0:
    #         return True
    #     return False

    # # 创建一个包含筛选后的数据项的子集
    # filtered_indices = [i for i, item in enumerate(dataset_exclude) if filter_data(item) and item > 50]
    # filtered_dataset = torch.utils.data.Subset(dataset_exclude, filtered_indices)
    
    # test_loader = torch.utils.data.DataLoader(
    #     filtered_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True
    # )

    
    
    # test_loader = torch.utils.data.DataLoader(
    #   Dataset(opt, 'val'), 
    #   batch_size=1, 
    #   shuffle=False,
    #   sampler=exclude_val_sampler,
    #   num_workers=1,
    #   pin_memory=True
    # )
    
    

    
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
    

    # L2_pixel_error,L2_mm_error,x_pixel_error,y_pixel_error = test(model, test_loader, opt)
    test(model, test_loader, opt)
    # print(L2_distance.shape)

    # print(f'The mean error distance (pixel): {L2_pixel_error:.2f}')
    # print(f'The mean error x (pixel): {x_pixel_error:.2f}')
    # print(f'The mean error y (pixel): {y_pixel_error:.2f}')
    # print(f'The mean error distance (mm): {L2_mm_error:.2f}')

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
