
import scipy.io
import numpy as np

import pandas as pd
import pathlib
from pathlib import Path
from PIL import Image
from pascal_voc_writer import Writer

import cv2 

import argparse
import pathlib

import pandas as pd
import tqdm
import os
import shutil

import json
import subprocess
import re

import h5py


EVE_info = []
def facial_landmarks_to_bbox(facial_landmarks):
    

    
    # 初始化边界框坐标
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    # # 计算最小和最大坐标值
    # for (x, y) in facial_landmarks:
    #     min_x = min(min_x, x)
    #     min_y = min(min_y, y)
    #     max_x = max(max_x, x)
    #     max_y = max(max_y, y)
        
    facial_landmarks_np =  np.array(facial_landmarks)
        
    # 使用NumPy计算最小和最大坐标值
    min_x = np.min(facial_landmarks_np[:, 0])
    min_y = np.min(facial_landmarks_np[:, 1])
    max_x = np.max(facial_landmarks_np[:, 0])
    max_y = np.max(facial_landmarks_np[:, 1])

    # 创建边界框
    bbox = (min_x, min_y, max_x, max_y)

    # 计算当前边界框的宽度和高度
    current_width = max_x - min_x
    current_height = max_y - min_y

    # 定义扩展的宽高比例
    expand_ratio = 1.2  # 例如，扩展 20%

    # 计算扩展后的宽度和高度
    expanded_width = current_width * expand_ratio
    expanded_height = current_height * expand_ratio

    # 计算新的边界框坐标，以保持中心不变
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    expanded_min_x = center_x - (expanded_width / 2)
    expanded_min_y = center_y - (expanded_height / 2)
    expanded_max_x = center_x + (expanded_width / 2)
    expanded_max_y = center_y + (expanded_height / 2)

    # 创建扩展后的边界框
    bbox = [int(expanded_min_x), int(expanded_min_y), int(expanded_max_x), int(expanded_max_y),int(center_x),int(center_y)]

    return bbox

def get_video_resolution(video_path):
    try:
    
        video_capture = cv2.VideoCapture(video_path)  

        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_capture.release()

        return width, height

    except IOError:
        print(f"無法開啟影像：{video_path}")
        return None

def create_directory_if_not_exists(path):
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    


def write_logger(filename, content):
    try:
        # 開啟檔案以寫入模式
        with open(filename, 'a') as file:  # 使用 'a' 模式以追加寫入方式打開檔案
            # 寫入內容
            file.write(content + "\n")  # 添加換行符號以區分每個內容
    except IOError:
        print("error")
        
        
def video2frames_xml(args, splited_set, video_folder, person_number, video_number, camera):

    splited_define = {"train":"1",
                      "val":"2",
                      "test":"3"}
    
    for splited_dict in splited_define.items():
        if splited_set == splited_dict[0]:
            splited_index = splited_dict[1]
            

    
    
    print(camera)
    cameras_define = {"basler":{
                    "index": "1",
                    "fps": "60"},
                "webcam_c":{
                    "index": "2",
                    "fps": "30"},
                "webcam_l":{
                    "index": "3",
                    "fps": "30"},
                "webcam_r":{
                    "index": "4",
                    "fps": "30"},}
    

    
            
    # print(camera)    
    # print(camera_index)
    # print(camera_fps)
        
    camera_h5  = str(camera) + ".h5"



            
    ##### h5 #####    
    for files_path in sorted(video_folder.glob('*')):        
        if camera_h5 in files_path.name:
            
            h5f = h5py.File(files_path, 'r')
            
            PoG_data = h5f['face_PoG_tobii/data'] 
            PoG_validity = h5f['face_PoG_tobii/validity']
            
            facial_landmarks_data = h5f['facial_landmarks/data'] 
            facial_landmarks_validity = h5f['facial_landmarks/validity']
            
            
            # camera_transformation_data = h5f['camera_transformation']
            
            head_rvec_data = h5f['head_rvec/data']
            
            head_tvec_data = h5f['head_tvec/data']
            
            # face_o_data = h5f['face_o/data']
            # face_o_validity = h5f['face_o/validity']
            
            face_R_data = h5f['face_R/data']
            face_R_validity = h5f['face_R/validity']
            
            
            face_g_data = h5f['face_g_tobii/data']
            face_g_validity = h5f['face_g_tobii/validity']
            
            face_h_data = h5f['face_h/data']
            face_h_validity = h5f['face_h/validity']
            
            
            
            
            splited_index_int = int(splited_index)
            person_number_int = int(person_number)
            video_number_int = int(video_number)
            # camera_index_int = int(camera_index)
            
            for index in range(len(PoG_data)):
                
                facial_landmarks = facial_landmarks_data[index]

                head_rvec = head_rvec_data[index]
                head_tvec = head_tvec_data[index]
                
                face_g =  face_g_data[index] 
                face_h =  face_h_data[index] 
                

                bbox = facial_landmarks_to_bbox(facial_landmarks)
                
                flattened_head_rvec = ' '.join([str(item[0]) for item in head_rvec])
                flattened_head_tvec = ' '.join([str(item[0]) for item in head_tvec])
                # flattened_face_g = ' '.join([str(item[0]) for item in face_g])
                flattened_face_g = ' '.join(map(str, face_g))
                # flattened_face_h = ' '.join([str(item[0]) for item in face_h])
                flattened_face_h = ' '.join(map(str, face_h))

   
                EVE_info.append(f"{flattened_head_rvec} {flattened_head_tvec} {flattened_face_g} {flattened_face_h} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}")
                
 

        

def ImageProcessing_EVE(dataset_dir, args):
    
    
    spilt_sets = ['train','val']
    for splited_set in spilt_sets: 
        for person_folder in sorted(dataset_dir.glob('*')):
            if splited_set in person_folder.name:
                
                
                person_number_find = re.search(r"(\d{2})", person_folder.name)
                person_number = person_number_find.group(1)
                
                for video_folder in sorted(person_folder.glob('*')):
                    if not os.path.isdir(video_folder):
                        continue
                    else:
                        print(video_folder)    
                         
                        video_number_find = re.search(r"(\d{3})", video_folder.name)
                        video_number = video_number_find.group(1)
                        
                    
                        
                        # camera = "webcam_c"
                        
                        # cameras_list = ["basler",
                        #     "webcam_c",
                        #     "webcam_l",
                        #     "webcam_r"]
                        
                        # cameras_list = [
                        #     "webcam_c",
                        #     "webcam_l",
                        #     "webcam_r"]
                        
                        cameras_list = [
                            "webcam_c"]
                        
                        for camera in cameras_list:
                        
                            video2frames_xml(args, splited_set, video_folder, person_number, video_number, camera)   
                        
                            
                                 


def main():
    output_txt_path = f'EVE_anno_gaze_sssss.txt'
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d','--datatype', type=str, required=True)
    # parser.add_argument('-s','--s_device', type=str)
    parser.add_argument('-o','--output_txt_path', type=str, required=True)
    
    args = parser.parse_args()
    output_txt_path = args.output_txt_path

     
     
    # abolute dir
    # dataset_dir = f'/home/owenserver/storage/Datasets/EVE/test_eve_dataset'
    # dataset_dir = f'/home/owenserver/storage/Datasets/EVE/eve_dataset'
    dataset_dir = f'/home/owenserver/storage/Datasets/EVE/test_s_eve_dataset'

    
    dataset_dir = pathlib.Path(dataset_dir)
    
  
    ImageProcessing_EVE(dataset_dir,args)
    
    
    with open(output_txt_path, 'w', encoding='utf-8') as output_txt:
        for info in EVE_info:
            output_txt.write(info + '\n')
    
        


if __name__ == '__main__':
    main()
    