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

import mediapipe as mp

from tkinter import *
from PIL import Image, ImageTk
import sys

def update_list_with_new_value(lst, new_value, k):
    if len(lst) < k:
        lst.insert(0, new_value)
    else:
        lst.pop()
        lst.insert(0, new_value)


def main(opt):
    
    
    print("TKinter")
    
    window = Tk()  #Makes main window
    window.overrideredirect(True)
    window.wm_attributes("-topmost", True)
    window.geometry("+600+200")
    display1 = Label(window)
    display1.grid(row=1, column=0, padx=0, pady=0)  #Display 1
    
    print("mediapipe function")
    
    mp_drawing = mp.solutions.drawing_utils                    # mediapipe 繪圖功能
    mp_selfie_segmentation = mp.solutions.selfie_segmentation  # mediapipe 自拍分割方法
    
    bg = cv2.imread('/home/owenserver/Python/CenterNet_gaze/src/tools/mpiifacegaze_eval/bg_green_640.jpg')   # 載入 windows 經典背
    
    print("mpiifacegaze eval")
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
    
    
    
    # mean = np.array([0.40789654, 0.44719302, 0.47026115],
    #                dtype=np.float32).reshape(1, 1, 3)
    # std  = np.array([0.28863828, 0.27408164, 0.27809835],
    #                dtype=np.float32).reshape(1, 1, 3)
    
    
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
    source = WebcamSource(camera_id=0 , width=640, height=360, fps=60, buffer_size=10)
    # window_height, window_width = 2360, 3840  # 設定螢幕大小
    window_height, window_width = 1080, 1920  # 設定螢幕大小
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
    
    
    # 定義不同按鍵對應的位置
    # key_positions = {
    #     ord('7'): (200, 100),ord('8'): (540, 100),ord('9'): (880, 100),
    #     ord('4'): (200, 360),ord('5'): (540, 360),ord('6'): (880, 360),
    #     ord('1'): (200, 660),ord('2'): (540, 660),ord('3'): (880, 660),
    #     ord('0'): (540, 360),
        
    # }
    
    key_positions = {
        ord('7'): (200, 100),ord('8'): (960, 100),ord('9'): (1720, 100),
        ord('4'): (200, 540),ord('5'): (960, 540),ord('6'): (1720, 540),
        ord('1'): (200, 940),ord('2'): (960, 940),ord('3'): (1720, 940),
        ord('0'): (960, 540),
        
    }
    
    
    WINDOW_NAME = 'data collection'
    
    monitor_mm = 480,260
    monitor_pixels = 1920, 1080
    
    # 初始化一个长度为 k 的列表
    k = 5
    moving_average_list_x = [960] * k
    moving_average_list_y = [540] * k
    


    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(WINDOW_NAME, 1080, 1920)

    
    
    
    fps_deque = collections.deque(maxlen=60)  # to measure the FPS
    prev_frame_time = 0
    # gaze_points = collections.deque(maxlen=64)
    gaze_points = collections.deque(maxlen=16)
    
    heads = {'hm': 1,'reg':2}


    head_conv = 64
    
    print('Creating model...')
    model = create_model(opt.arch, heads, head_conv)
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_pl001_2/model_2.pth"
    
    model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/mpiifacegaze/resdcn_18/gaze_resdcn18_csp_kr_resize_petrain_eve/gaze_resdcn18_csp_kr_resize_p14_petrain_eve/model_16.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_3_webcam/model_3.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/mpiifacegaze/resdcn_18/gaze_resdcn18_ep70_all_keep_res_resize_pl001_p12_pl_fix/model_70.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_pl001_1/model_1.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/eve/resdcnface_18/gaze_eve_resdcnface_18_eve_ep10_f10/model_2.pth"
    model = load_model(model, model_path)
    
    model.eval()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    
    model = model.to(device)
        
    
    
    for frame_idx, img in enumerate(source):
        # height, width, _ = img.shape
        # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image_rgb = cv2.flip(frame, 1)
        
        # img = cv2.flip(img, 1)
    

        
        # img = cv2.resize(img, (opt.resize_raw_image_w, opt.resize_raw_image_h), interpolation=cv2.INTER_LINEAR)
        
        print(img.shape)
        
        # with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
            
        #     img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        #     results = selfie_segmentation.process(img2)   # 取得自拍分割結果
        #     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1 # 如果滿足模型判斷條件 ( 表示要換成背景 )，回傳 True
        #     img = np.where(condition, img, bg)
        
        
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
            
        # print("input_h , input_w = ", input_h,input_w )
            
            
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, 
                        (input_w, input_h),
                        flags=cv2.INTER_LINEAR)
    
        # print("inp.shape =", inp.shape)

        inp = (inp.astype(np.float32) / 255.)
        # if self.split == 'train' and not self.opt.no_color_aug:
        #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        
        inp = (inp - mean) / std
        
        # print("inp_output.shape =", inp.shape)
        inp_output = inp
        
        inp = inp.transpose(2, 0, 1)
        
        print("img.shape =", img.shape)
        
        inp = torch.from_numpy(inp).unsqueeze(0)
        

        inp = inp.to(device)
        
        ################ vp  ################
        vp_width, vp_height = opt.vp_w, opt.vp_h
        # vp = np.zeros((vp_height, vp_width , 1), dtype=np.float32)
        # vp = vp.transpose(1,2,0)
        
        

        
        vp_c = np.array([ vp_width / 2.0, vp_height/ 2.0], dtype=np.float32)
        if opt.keep_res:
            # trans_output_h = (vp_height | self.opt.pad) + 1
            # trans_output_w = (vp_width | self.opt.pad) + 1
            # vp_s = np.array([trans_output_w, trans_output_h], dtype=np.float32)
            trans_output_h = (vp_height | opt.pad) + 1
            trans_output_w = (vp_width | opt.pad) + 1
            vp_s = np.array([vp_width, vp_height], dtype=np.float32)
        # else:
        #     vp_s = max(vp_width, vp_height) * 1.0
        
        
        
        
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
        
        hm_squeezed = new_hm_out.squeeze(dim=1)
        # hm_numpy = hm_squeezed.detach().numpy()
        hm_squeezed = hm_squeezed.detach().numpy()
        
        # 將灰度值縮放到0-255之間
        hm_numpy = ((hm_squeezed - hm_squeezed.min()) / (hm_squeezed.max() - hm_squeezed.min()) * 255).astype(np.uint8)[0] 
        

        
        # hm_numpy = (hm_squeezed.detach().numpy()  * 255).astype(np.uint8)[0] 
        # hm_numpy = (hm_squeezed.detach().numpy() * 255).astype(np.uint8)

        max_index = np.argmax(hm_numpy)
        # 將索引轉換為二維坐標
        max_coordinates = np.unravel_index(max_index, hm_numpy.shape)

        print("hm_out max : ",max_index)
        print("Max Value:", hm_numpy[max_coordinates])
        print("Max Coordinates (x, y):", max_coordinates)
        
        # print("hm_out max : ",max_index)
        
        
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
        dets_org_coord = dets_out
        print(f"dets_out: {dets_out}")
        
        # cls_id = 1
        # dets_org_coord = torch.tensor(dets_out[0][cls_id][:])
        # dets_org_coord = dets_org_coord[:,:2] 
        print(f"dets_org_coord: {dets_org_coord}")
        
        

        # dets_modified = int(dets_org_coord[:,0]-960) , int(dets_org_coord[:,1] - (1080))
        # dets_modified = dets_org_coord - torch.tensor([1200, 1800])
        
        # dets_modified = (int((max_coordinates[1]-41)*1080/168),int((max_coordinates[0]-52)*720/96))
        # dets_modified = (int((max_coordinates[1]-48)*1920/84),int((max_coordinates[0]-48)*1080/48))
        
        # dets_modified = (int((max_coordinates[1]-48)*1920/84),int((max_coordinates[0]-42)*1080/48))
        
        # dets_modified = (int((max_coordinates[1]-60)*1080/47),int((max_coordinates[0]-67)*720/29))
        # dets_modified = dets_org_coord * scale_factor_y
        
        
        # dets_modified = (int((dets_org_coord[0][0]-960)),int((dets_org_coord[0][1]-1080)))
        # dets_modified = (int((dets_org_coord[0][0]-1260)),int((dets_org_coord[0][1]-980)))
        dets_modified = (int((dets_org_coord[0][0]-960)),int((dets_org_coord[0][1]-980)))
        # dets_modified = (int((dets_org_coord[0][0]-960)),int((dets_org_coord[0][1]-1180)))
        
        
        # if dets_modified[0] > 1920 or dets_modified[0] < 0 or dets_modified[1] > 1080  or dets_modified[1] < 0 : 
        #     # eliminate out of scope
        #     continue
        
        print("dets_modified: " , dets_modified)
        calib_coord = dets_modified
        # coord = dets_modified.to(torch.int)
            
        
        # coord = dets_org_coord[:,0]-960 , dets_org_coord[:,1] - (1080)
        # coord = dets_org_coord[:,0]-1200 , dets_org_coord[:,1] - (1200)
        # print("coord: " , coord)
        
        # coord_x, coord_y = int(coord[0].item()*scale_factor_x), int(coord[1].item()*scale_factor_y)
        
        # x_value = int(coord[0][0].item())
        # y_value = int(coord[0][1].item())
        
        # # calib_coord = int(dets_modified[0].item()), int(dets_modified[1].item())
        # calib_coord = (x_value,y_value)
        # calib_coord = (int(coord[0].item()), int(coord[1].item()))
        # print("calib coord: " , calib_coord)
        
        # cv2.circle(black_background, (coord_x, coord_y), radius, red_color, thickness)
        
        # point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, calib_coord)
        update_list_with_new_value(moving_average_list_x, calib_coord[0], k)
        update_list_with_new_value(moving_average_list_y, calib_coord[1], k)
        calib_coord = (int(sum(moving_average_list_x) / len(moving_average_list_x)),int(sum(moving_average_list_y) / len(moving_average_list_y)))
        point_on_screen  = calib_coord
        gaze_points.appendleft(point_on_screen)
        print("point_on_screen: " , point_on_screen)
        black_background.fill(0)
        
        # points_np = point_on_screen.numpy()
        
        # for point in points_np:
        #     cv2.circle(black_background, tuple(point), 5, (0, 0, 255), -1)  # 半徑為5，顏色為紅色 (BGR 格式)
        
        cv2.circle(black_background, (refpoint_x,refpoint_y), 10, (0,165,255), -1)  # 繪製紅色圓圈
        
        
        PoG_error = (int(np.linalg.norm(np.array((refpoint_x, refpoint_y)) - np.array(point_on_screen))) ,(refpoint_x - point_on_screen[0]),(refpoint_y - point_on_screen[1]))
        position = (refpoint_x -75 , refpoint_y -25 )
        cv2.putText(black_background, str(PoG_error), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        

        print("gaze_points", len(gaze_points))

        for idx in range(1, len(gaze_points)):
            thickness = round((len(gaze_points) - idx) / len(gaze_points) * 5) + 1
            cv2.line(black_background, gaze_points[idx - 1], gaze_points[idx], (0, 0, 255), thickness)
            
            # PoG_error = ((refpoint_x - gaze_points[idx][0]),(refpoint_y - gaze_points[idx][1]))
            # position = (refpoint_x -50 , refpoint_y -50 )
            # cv2.putText(black_background, str(PoG_error), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            
            # cv2.circle(black_background, gaze_points[idx], radius, red_color, thickness)
            
        # cv2.imshow(window_normal, black_background)
        if frame_idx % 3 == 0:
            
            # PoG_error = ((refpoint_x - gaze_points[idx][0]),(refpoint_y - gaze_points[idx][1]))
            # position = (refpoint_x -50 , refpoint_y -50 )
            # cv2.putText(black_background, str(PoG_error), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            
            cv2.namedWindow(window_normal, flags=cv2.WINDOW_GUI_NORMAL)
            
            cv2.imshow(window_normal, black_background)
            pass
        
        # print("calib coord: " , coord)
        
    
        # inp_output
        cv2.imshow("inp_output", inp_output)
        # inp = inp.transpose(2, 0, 1)
        
        # cv2.namedWindow('Tracking', flags=cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Tracking", img)
        # cv2.imshow('Heatmap Black and White', hm_numpy, cmap='gray')
        # cv2.imshow('Heatmap', hm_numpy)
        
        
    
        new_frame_time = time.time()
        fps_deque.append(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        if frame_idx % 60 == 0:
            print(f'FPS: {np.mean(fps_deque):5.2f}')
            
        
        key = cv2.waitKey(1) & 0xFF
            
        if key in key_positions:
            refpoint_x, refpoint_y = key_positions[key]
        elif key == ord("q"):
            break

        # # If the 'q' key is pressed, stop the loop
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    # Cleanup the camera and close any open windows
    source.release()
    cv2.destroyAllWindows()
    

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
