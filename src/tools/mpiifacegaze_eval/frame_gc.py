#!/usr/bin/env python
import os
import pathlib
import math

import collections
import time
from argparse import ArgumentParser
from mpiifacegaze_eval_lib.webcam import WebcamSource
from mpiifacegaze_eval_lib.utils_camera import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen
import cv2

import matplotlib.pyplot as plt
import glob
import re

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
from mpiifacegaze_eval_lib.utils.image import get_affine_transform, affine_transform, transform_preds


def update_list_with_new_value(lst, new_value, k):
    if len(lst) < k:
        lst.insert(0, new_value)
    else:
        lst.pop()
        lst.insert(0, new_value)
        
def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group()) if match else float('inf')


def main(opt):
    
    print("mpiifacegaze eval")
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    # mean = np.array([0.485, 0.456, 0.406],
    #                dtype=np.float32).reshape(11, 1, 3)
    # std  = np.array([0.229, 0.224, 0.225],
    #                dtype=np.float32).reshape(1, 1, 3)
    
    mean = np.array([0.5893, 0.5006, 0.4467],
                   dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.2713, 0.2683, 0.2581],
                   dtype=np.float32).reshape(1, 1, 3)
    
    
    
    

    
    # camera = cv2.VideoCapture(0)
    # while True:
    #     # Grab the current paintWindow
    #     (grabbed, frame) = camera.read()
    #     frame = cv2.flip(frame, 1)
    #     print()
        
        
    #     # Show the frame and the paintWindow image
    #     cv2.imshow("Tracking", frame)
    #     # cv2.imshow("Paint", paintWindow)

    #     # If the 'q' key is pressed, stop the loop
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    # # Cleanup the camera and close any open windows
    # camera.release()
    # cv2.destroyAllWindows()
    
    # setup webcam
    # source = WebcamSource(width=1280, height=720, fps=60, buffer_size=10)
    # source = WebcamSource(camera_id=0 , width=640, height=640, fps=60, buffer_size=10)
    # window_height, window_width = 2360, 3840  # 設定螢幕大小
    window_height, window_width = 640, 640  # 設定螢幕大小
    # window_height, window_width = 720, 1080  # 設定螢幕大小
    black_background = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    
    
    
    window_normal = 'Black Window'
    # cv2.namedWindow(window_normal, cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow(black_background, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(black_background, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    
    
    # virtual_screen_size = "1920x1080"
    # os.environ['DISPLAY'] = f":99.0"
    # os.system(f"Xvfb {os.environ['DISPLAY']} -screen 0 {virtual_screen_size}x24 &")
    
    # cv2.namedWindow('Red Point on Black Background', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Red Point on Black Background', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    
    # 在黑色背景上畫一個紅色的點
    red_color = (0, 0, 255)  # BGR format, 紅色
    radius = 5  # 點的半徑
    thickness = -1  # -1 表示填充整個圓
    refpoint_x, refpoint_y = 960,540
    

    
    
    
    WINDOW_NAME = 'data collection'
    



    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(WINDOW_NAME, 1080, 1920)

    

    
    heads = {'hm': 1,'reg':2}


    head_conv = 64
    
    print('Creating model...')
    
    arch = 'resdcn_18'
    model = create_model(arch, heads, head_conv)
    
    model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/gazecapture/resdcn_18/gaze_gazecapture_all_no_scp_pl01/model_58.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_pl001_2/model_5.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_3_webcam/model_3.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/mpiifacegaze/resdcn_18/gaze_resdcn18_ep70_all_keep_res_resize_pl001_p12_pl_fix/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_pl001_1/model_1.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/eve/resdcnface_18/gaze_eve_resdcnface_18_eve_ep10_f10/model_2.pth"
    model = load_model(model, model_path)
    
    
    # folder_path = '/home/owenserver/storage/Datasets/GazeCapture/00002_train_phone_s/'
    folder_path = '/home/owenserver/storage/Datasets/GazeCapture/00028_train_tablet_s/'
    # folder_path = '/home/owenserver/storage/Datasets/GazeCapture/00110_test_phone_s/'
    # folder_path = '/home/owenserver/storage/Datasets/GazeCapture/00010_test_tablet_s/'




    # 使用glob模塊列出文件夾中的所有圖像文件
    image_files = glob.glob(folder_path + '*.jpg')
    image_files = sorted(image_files, key=extract_number)
    
    
    
    img_list = []
    
    
    model.eval()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda")
    # device = torch.device(opt.device)
    
    model = model.to(device)
    
    
    # 00002_train_phone_s
    # screenWH = np.array([[320 , 568],
    #         [320 , 568],
    #         [320 , 568],
    #         [568 , 320],
    #         [568 , 320]])
    
    # PoG = np.array([[160 , 284],
    #         [279 , 64],
    #         [280 , 528],
    #         [304 , 269],
    #         [164 , 153]])
    
    
    # 00028_train_tablet_s
    screenWH = np.array([[768, 1024],
            [768, 1024],
            [768, 1024]])
    
    PoG = np.array([[468 , 252],
            [87 , 166],
            [299 , 631]])
    
    
    # 00110_test_phone_s
    # screenWH = np.array([[320, 568],
    #         [320, 568],
    #         [320, 568]])
    
    # PoG = np.array([[100 , 284],
    #         [74 , 138],
    #         [220 , 406]])
    
    
    # 00010_test_tablet_s
    # screenWH = np.array([[768, 1024],
    #         [768, 1024],
    #         [768, 1024]])
    
    # PoG = np.array([[100 , 284],
    #         [74 , 138],
    #         [220 , 406]])
    
    

    

    
    
    for idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        print("img_path = ", img_path)
        if img is not None:
            img_list.append((img, img_path , screenWH[idx] , PoG[idx]))
        else:
            print(f"無法讀取圖像：{img_path}")
            
    print("-------------------------------------")
    
    for img, img_path, screenWH , PoG in img_list:
        # print(img_path)
        # height, width, _ = img.shape
        # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image_rgb = cv2.flip(frame, 1)
        
        # img = cv2.flip(img, 1)
        # img = cv2.resize(img, (opt.resize_raw_image_w, opt.resize_raw_image_h), interpolation=cv2.INTER_LINEAR)
        
        # cv2.imshow('image', img)
        # cv2.waitKey(0)  # 等待按鍵按下
        
        
        # height, width, _ = img.shape
        # print("img.shape = ", img.shape )
        
        img_height, img_width = img.shape[0], img.shape[1]
        input_h, input_w = opt.input_h, opt.input_w
        
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if opt.keep_res:
            input_h = (img_height | opt.pad) + 1
            input_w = (img_width | opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img_width, img_height) * 1.0
            input_h, input_w = opt.input_h, opt.input_w
            
        print("input_h , input_w = ", input_h,input_w )
            
            
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, 
                        (input_w, input_h),
                        flags=cv2.INTER_LINEAR)
    

        inp = (inp.astype(np.float32) / 255.)
        # if self.split == 'train' and not self.opt.no_color_aug:
        #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        
        inp = (inp - mean) / std
        
        inp_output = inp
        
        inp = inp.transpose(2, 0, 1)
        
        inp = torch.from_numpy(inp).unsqueeze(0)

        inp = inp.to(device)
        
        ################ vp  ################
        vp_width, vp_height = opt.vp_w, opt.vp_h
        # vp = np.zeros((vp_height, vp_width , 1), dtype=np.float32)
        # vp = vp.transpose(1,2,0)
        
        ############ check  sc_point ############
        
        sc_gazepoint = PoG
        sc_width = screenWH[0] 
        sc_height = screenWH[1]
        camera_screen_x_offset = 0 
        camera_screen_y_offset = 0
        vp_gazepoint = np.array([(vp_width/2)+(sc_gazepoint[0]-(sc_width/2))+ camera_screen_x_offset, (vp_height/2)+(sc_gazepoint[1]-(sc_height/2))+camera_screen_y_offset], dtype=np.float32)
        
        print("vp_gazepoint =",vp_gazepoint)
        
        vp_c = np.array([ vp_width / 2.0, vp_height/ 2.0], dtype=np.float32)
        if opt.keep_res:
            # trans_output_h = (vp_height | self.opt.pad) + 1
            # trans_output_w = (vp_width | self.opt.pad) + 1
            # vp_s = np.array([trans_output_w, trans_output_h], dtype=np.float32)
            trans_output_h = (vp_height | opt.pad) + 1
            trans_output_w = (vp_width | opt.pad) + 1
            vp_s = np.array([vp_width, vp_height], dtype=np.float32)
        else:
            # vp_s = max(vp_width, vp_height) * 1.0
            vp_s = np.array([vp_width, vp_height], dtype=np.float32)
        
        
        
        
        ################ model  ################
        
        outputs = model(inp)
        
        output = outputs[-1]
        hm = output['hm'].sigmoid_()
        hm_out = output['hm'].sigmoid_().cpu()
        

        x_start, y_start = 52, 41
        x_end, y_end = x_start + 44, y_start + 86

        # 創建一個相同形狀但數值全為零的張量

        new_hm_out = torch.zeros_like(hm_out)

        # 在指定區域內保留原來的數值
        new_hm_out[0, 0, x_start:x_end, y_start:y_end] = hm_out[0, 0, x_start:x_end, y_start:y_end]

        
        reg = output['reg']
        
        
        # hm mask
        

        
        dets = ctdet_gaze_decode(hm, reg=reg, K=opt.K)
        # dets = ctdet_gaze_decode(hm, reg=reg, K=5)
        
        # hm_squeezed = new_hm_out.squeeze(dim=1)
        # # hm_numpy = hm_squeezed.detach().numpy()
        # hm_squeezed = hm_squeezed.detach().numpy()
        
        # # 將灰度值縮放到0-255之間
        # hm_numpy = ((hm_squeezed - hm_squeezed.min()) / (hm_squeezed.max() - hm_squeezed.min()) * 255).astype(np.uint8)[0] 
        

        
        # # hm_numpy = (hm_squeezed.detach().numpy()  * 255).astype(np.uint8)[0] 
        # # hm_numpy = (hm_squeezed.detach().numpy() * 255).astype(np.uint8)

        # max_index = np.argmax(hm_numpy)
        # # 將索引轉換為二維坐標
        # max_coordinates = np.unravel_index(max_index, hm_numpy.shape)

        # print("hm_out max : ",max_index)
        # print("Max Value:", hm_numpy[max_coordinates])
        # print("Max Coordinates (x, y):", max_coordinates)
        
        # # print("hm_out max : ",max_index)
        
        
        # detections = torch.cat([gp, scores, clses], dim=2)
        # print(f"dets: {dets}")
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        
        print("dets:", dets[0][0][0], "  ",dets[0][0][1] )
        
        # print("vp_c , vp_s = ", vp_c,vp_s)
        

        
        # output['hm'].shape[2]  H 
        # output['hm'].shape[3]  W
        print("output['hm'].shape[2] /  output['hm'].shape[3] : ", output['hm'].shape[2], "   ", output['hm'].shape[3]  )
        print("vp_c /  vp_s : ", vp_c, "   ", vp_s  )
        
        # dets_out = ctdet_gaze_post_process(
        #     dets.copy(), vp_c,
        #     vp_s,
        #     output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        # dets[i, :, 0:2], c[i], s[i], (w, h)
        # cords = np.array(dets[0][0][0],dets[0][0][1])
        # output_s = np.array(output['hm'].shape[2], output['hm'].shape[3])
        dets_out = transform_preds(dets[0, :, 0:2], 
                                   vp_c,vp_s,
                                   (output['hm'].shape[3], output['hm'].shape[2]))
        
        # dets_out = transform_preds(dets[0, :, 0:2], 
        #                     vp_c,vp_s,
        #                     (output['hm'].shape[3], output['hm'].shape[2]))
        # width / height
        dets_org_coord = dets_out
        print(f"dets_out: {dets_out}")
        
        # cls_id = 1
        # dets_org_coord = torch.tensor(dets_out[0][cls_id][:])
        # dets_org_coord = dets_org_coord[:,:2] 
        print(f"dets_org_coord: {dets_org_coord}")
        
        

        # vp_gazepoint = np.array([(vp_width/2)+(sc_gazepoint[0]-(sc_width/2))+ camera_screen_x_offset, (vp_height/2)+(sc_gazepoint[1]-(sc_height/2))+camera_screen_y_offset], dtype=np.float32)

        # (sc_width/2) = (1920 /2) = 960
        # (vp_width/2) = (3840 /2) = 1920 
        # (sc_height/2) = (1080 /2) = 540
        # (vp_height/2) = (2360 /2) = 1180 
        
        dets_modified_vp = (int((vp_gazepoint[0] + (sc_width/2) - (vp_width/2))),int((vp_gazepoint[1]+ (sc_height/2)-(vp_height/2))))
        
        print("dets_modified_vp: " , dets_modified_vp)
        
        dets_modified = (int((dets_org_coord[0][0] + (sc_width/2) - (vp_width/2))),int((dets_org_coord[0][1]+ (sc_height/2)-(vp_height/2))))
        # dets_modified = (int((dets_org_coord[0][0]-960)),int((dets_org_coord[0][1]-1180)))
        
        
        print("dets_modified: " , dets_modified)
        
        # PoG = int(PoG)
        
        PoG = PoG.astype(int)
        print("GT : " , PoG[0],PoG[1])

        error = (dets_modified - PoG)
        error = np.round(error,2)
        euc_error = math.sqrt(error[0]**2 + error[1]**2)
        euc_error = round(euc_error,2)
        print(" x y error : " , error)
        print(" euc_error : " , euc_error)
        
        # eud distance , x , y
        PoG_error = (error , euc_error)

        black_background.fill(0)
        error_text_position =  (100, 100)
        cv2.putText(black_background, str(PoG_error), error_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        


        # cv2.putText(black_background, str(PoG_error), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        yellow_color = (0, 255, 255)  # BGR format, yellow 
        
        cv2.circle(black_background, dets_modified, radius, yellow_color, thickness)
        
        cv2.circle(black_background, PoG, radius, red_color, thickness)
        # cv2.circle(black_background, dets_modified, radius, point_color, thickness)
        
        cv2.namedWindow(window_normal, flags=cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(window_normal, black_background)

            

        
        # cv2.imshow("inp_output", inp_output)
        # cv2.imshow("Tracking", img)

        
     
        
        key = cv2.waitKey(0) & 0xF
        
        
        if key == ord('q'):
            break
        
        cv2.imshow("Tracking", img)
        cv2.destroyAllWindows()

    # Cleanup the camera and close any open windows
    # source.release()
    # cv2.destroyAllWindows()
    

if __name__ == '__main__':
    opt = opts().parse()
    # parser = ArgumentParser()
    # parser.add_argument("--calibration_matrix_path", type=str, default='./calibration_matrix.yaml')
    # parser.add_argument("--model_path", type=str, default='./p00.ckpt')
    # parser.add_argument("--monitor_mm", type=str, default=None)
    # parser.add_argument("--monitor_pixels", type=str, default=None)
    # parser.add_argument("--visualize_preprocessing", type=bool, default=False)
    # parser.add_argument("--visualize_laser_pointer", type=bool, default=True)
    # parser.add_argument("--visualize_3d", type=bool, default=False)
    # args = parser.parse_args()

    # if args.monitor_mm is not None:
    #     args.monitor_mm = tuple(map(int, args.monitor_mm.split(',')))
    # if args.monitor_pixels is not None:
    #     args.monitor_pixels = tuple(map(int, args.monitor_pixels.split(',')))
  
  
    main(opt)
