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
    losses_gaze = AverageMeter()
    update_heatmap = UpdateHeatmap(opt.vp_w,opt.vp_h)
    
    L2_pixel_errors_list = []

    # predictions = []
    # gts = []
    with torch.no_grad():
        for iter_id, batch in enumerate(test_loader):
            print(f"Iteration {iter_id}/ {len(test_loader)}",end="\r")
            
            if iter_id > 3000:
                break
            
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True) 
            outputs = model(batch['input'])
            # print(f"shape of input {batch['input'].shape}")
            
            batch_image = batch['input']
            
            output_image = batch_image[0].detach().cpu().numpy()
            output_image = output_image.transpose(1, 2, 0)
            # plt.imshow(output_image, cmap="gray")
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            
            mm_per_pixel = torch.tensor(batch['meta']['mm_per_pixel'].numpy())
            
            # print(type(mm_per_pixel))
            # mm_per_pixel = torch.tensor(batch['meta']['mm_per_pixel'].numpy())
            # mm_per_pixel = mm_per_pixel.view(N, 1).expand(N, 2)
            # mm_per_pixel = float(mm_per_pixel.detach().cpu().numpy())
            # print("mm_per_pixel",f'{mm_per_pixel}')
            
            img_id = batch['meta']['img_id']
            # print(f"img_id : {img_id}")
            # print(output)
            output = outputs[-1]
            hm = output['hm'].sigmoid_()
            
            
            
            # heatmap Ground Truth vs Predict
            gt_hm = batch['hm'][0].detach().cpu().numpy()
            gt_hm = gt_hm.transpose(1, 2, 0)
    
            output_hm = output['hm'][0].detach().cpu().numpy()
            output_hm = output_hm.transpose(1, 2, 0)
            
            # plt.subplot(1, 2, 1)
            # plt.imshow(gt_hm, cmap="gray")
            # plt.title('Ground Truth')

            # plt.subplot(1, 2, 2)
            # plt.imshow(output_hm, cmap="gray")
            # plt.title('Predict')
            # plt.show()
            
            
            
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
            
            L2_mm_error = torch.sqrt(error*(mm_per_pixel**2))
            L2_mm_error = L2_mm_error.mean()
            L2_mm_errors.update(L2_mm_error)
            
            
            # unit_error_display(dets_gt_org_coord,error)
            
            x = int(dets_gt_org_coord[0][0])
            y = int(dets_gt_org_coord[0][1])
            # print("dets_gt_org_coord: ", x,"  ", y )
            
            
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
    plt.show()
    
    
    

    mean_L2_pixel_errors = L2_pixel_errors.avg
    mean_L2_mm_errors = L2_mm_errors.avg
    
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
    plt.show()
    
    



            
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
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/all_csp_kr_resize_pl/pl001/gaze_resdcn18_ep70_all_csp_kr_resize_pl001_p10/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/cross_baseline_sp_norm_flipfix/gaze_resdcn18_ep70_all_base_sp_norm_flipfix_p02/model_70.pth"
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/cross_baseline_sp_norm_gp_shfit/gaze_resdcn18_ep70_all_base_sp_norm_gp_shift_p08/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcn18_ep70_all_norm_csp_kr_pl001_p05_pl_fix/model_70.pth"
    
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep140_test/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep30_test_all/model_30.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep70_all_phone_pl01/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_gazecapture_ep70_all/model_70.pth"
    
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcncut_18_mpii_r10_ep70_all_norm_csp_kr_pl001_p05/model_14.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_resdcncut_18_mpii_r10_ep70_all_norm_csp_kr_pl001_p05/model_14.pth" 
    
    
    model_path = "/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/gaze_eve_sc_kr/model_5.pth"
    
    
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
