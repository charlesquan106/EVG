
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
    bbox = [int(expanded_min_x), int(expanded_min_y), int(expanded_max_x), int(expanded_max_y)]

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
        
        
def video2frames_xml(args, splited_set, video_folder, output_img_dir, output_xml_dir, person_number, video_number, camera):

    splited_define = {"train":"1",
                      "val":"2",
                      "test":"3"}
    
    for splited_dict in splited_define.items():
        if splited_set == splited_dict[0]:
            splited_index = splited_dict[1]
            

    
    
    
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
    

    
    for camera_dict in cameras_define.items():
        if camera == camera_dict[0]:
            camera_index = camera_dict[1]["index"]
            camera_fps = camera_dict[1]["fps"]
            
            # print(camera)    
            # print(camera_index)
            # print(camera_fps)
            
            
            
    # print(camera)    
    # print(camera_index)
    # print(camera_fps)
        
    camera_mp4 = str(camera) + ".mp4"
    camera_h5  = str(camera) + ".h5"

    no_images = args.no_images 

    ##### mp4 #####
    for files_path in sorted(video_folder.glob('*')):
        # print(files_path)
        
        if camera_mp4 in files_path.name :
            # print(files_path)
            
            
            # get_video_resolution 
            video_path = str(files_path)
            img_width, img_height = get_video_resolution(video_path)
            
            output_image_dir = output_img_dir
            
            splited_index_int = int(splited_index)
            person_number_int = int(person_number)
            video_number_int = int(video_number)
            camera_index_int = int(camera_index)
            
            image_format = "%05d.jpg"
            
            img_filename = f"{splited_index_int:01d}{person_number_int:02d}{video_number_int:03d}{camera_index_int:02d}{image_format}"

            # params= f"-vf fps={camera_fps}"
            params= f"-f image2"
            # params = " -f image2 -vsync vfr -q:v 2"
            cmd = f"ffmpeg -i {files_path} {params} {output_image_dir}/{img_filename}"
            # print("cmd: ", cmd)
            if no_images != True :
                subprocess.run(cmd,shell= True, stderr=subprocess.PIPE,stdout=subprocess.DEVNULL)
            
    ##### h5 #####    
    for files_path in sorted(video_folder.glob('*')):        
        if camera_h5 in files_path.name:
            
            h5f = h5py.File(files_path, 'r')
            
            PoG_data = h5f['face_PoG_tobii/data'] 
            PoG_validity = h5f['face_PoG_tobii/validity']
            
            facial_landmarks_data = h5f['facial_landmarks/data'] 
            facial_landmarks_validity = h5f['facial_landmarks/validity']
            
            
            camera_transformation_data = h5f['camera_transformation']
            
            head_rvec_data = h5f['head_rvec/data']
            
            face_o_data = h5f['face_o/data']
            face_o_validity = h5f['face_o/validity']
            
            face_R_data = h5f['face_R/data']
            face_R_validity = h5f['face_R/validity']
            
            
            
            
            splited_index_int = int(splited_index)
            person_number_int = int(person_number)
            video_number_int = int(video_number)
            camera_index_int = int(camera_index)
            
            
            for index in range(len(PoG_data)):
                index_plusone = index+1
                
                img_filename = f"{splited_index_int:01d}{person_number_int:02d}{video_number_int:03d}{camera_index_int:02d}{index_plusone:05d}.jpg"
                output_image_path = output_image_dir / Path(img_filename)
                
                xml_filename = f"{splited_index_int:01d}{person_number_int:02d}{video_number_int:03d}{camera_index_int:02d}{index_plusone:05d}.xml"
                output_xml_path = output_xml_dir / Path(xml_filename)
                # if not PoG_validity[index] or not facial_landmarks_validity[index]:
                #     print("remove no validity image and xml ", output_image_path.name)
                #     os.remove(output_image_path)
                #     continue  
                x_px,y_px = PoG_data[index]
                facial_landmarks = facial_landmarks_data[index]
                camera_transformation = camera_transformation_data[:]
                head_rvec = head_rvec_data[index]
                face_R = face_R_data[index]
                face_o = face_o_data[index]
                
                # print(type(output_image_path))
                # print(output_image_path)
                bbox = facial_landmarks_to_bbox(facial_landmarks)
                
                # eye
                reye = []
                reye_x = int((facial_landmarks[36][0]+facial_landmarks[39][0])/2)
                reye_y = int((facial_landmarks[36][1]+facial_landmarks[39][1])/2)
                reye = [reye_x,reye_y]
                
                leye = []
                leye_x = int((facial_landmarks[42][0]+facial_landmarks[45][0])/2)
                leye_y = int((facial_landmarks[42][1]+facial_landmarks[45][1])/2)
                leye = [leye_x,leye_y]
                
                
                #**************** check bbox on image 
                # img = cv2.imread(str(output_image_path))
                
                # # cv2.imshow('Image', img)
    
                # # # 等待用户按下任意键后关闭窗口
                # # cv2.waitKey(0)
                
                # img_black = np.zeros_like(img)
                # print(bbox)

                # # img_black[bbox[0]:bbox[2], bbox[1]:bbox[3]] = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # img_black[bbox[1]:bbox[3], bbox[0]:bbox[2]] = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # img = img_black

                # cv2.imwrite(str(output_image_path), img_black)
                
                #**************** check bbox on image 
                
                
                
                camera_dict = {
                        'cameraMatrix': [],
                        'distCoeffs': []
                        }
                monitorPose_dict = {
                        'rvects': [],
                        'tvecs': []
                        }
                screenSize_dict = {
                        'unit_mm': [],
                        'unit_pixel': []
                        }
                
                
                unit_mm = []
                unit_pixel = []
                
                # 1920 x 1080  (25 inch) / 553 x 311 (mm)
                ScreenHeightMm = 311
                ScreenWidthMm = 553
                ScreenHeightPixel = 1080
                ScreenWidthPixel = 1920
                
                unit_mm.append(ScreenWidthMm)
                unit_mm.append(ScreenHeightMm)

                unit_pixel.append(ScreenWidthPixel)
                unit_pixel.append(ScreenHeightPixel)
                

                
                screenSize_dict['unit_mm'] = unit_mm
                screenSize_dict['unit_pixel'] = unit_pixel
                
                
                rvects = []
                tvecs = []
                
                
                monitorPose_dict['rvects'] = rvects
                monitorPose_dict['tvecs'] = tvecs
                
                # create pascal voc writer (image_path, camera_dict, monitorPose_dict, width, height)
                writer = Writer(output_image_path, img_width, img_height, 'EVE')
                
                
                # gaze_origin = [row.gaze_origin_x,row.gaze_origin_y,row.gaze_origin_z]
                # gaze_target = [row.gaze_target_x,row.gaze_target_y,row.gaze_target_z]
                gaze_origin = face_o
                
                gaze_target = []
                
                face_bbox = [] 
                
                face_bbox = bbox
                # print(type(camera_transformation))
                
                camera_transformation = camera_transformation.flatten()
                
                head_rvec = head_rvec.flatten()
                
                face_R = face_R.flatten()
                # print(head_rvec.shape)
                # print(head_rvec)

            
            
                ##### write to xml #####  
            
                # writer.addGaze(int(x_px), int(y_px) , gaze_origin, gaze_target\
                #     , camera_dict, monitorPose_dict, screenSize_dict, face_bbox)
                
                
                # writer.addGaze_ld(int(x_px), int(y_px) , gaze_face_origin, gaze_target\
                #     , camera_dict, monitorPose_dict, screenSize_dict, face_bbox, reye, leye)
                
                                           
                                
                writer.addGaze_ld_eve(int(x_px), int(y_px) , gaze_origin, gaze_target\
                    , camera_dict, camera_transformation , head_rvec, face_R
                    , monitorPose_dict, screenSize_dict, face_bbox, reye, leye)
                
                ##### write to xml #####  
                writer.save(output_xml_path)
                
                if not PoG_validity[index] or not facial_landmarks_validity[index] or not face_o_validity[index] or not face_R_validity[index]:
                    print("remove no validity image and xml ", output_image_path.name)
                    
                    if os.path.exists(output_image_path):
                        os.remove(output_image_path)
                    if os.path.exists(output_xml_path):
                        os.remove(output_xml_path)
                    continue 
                
                

            

        
                
                
        

def ImageProcessing_EVE(dataset_dir, output_dir, args):
    
    spilt_sets = args.datatype 
    print(spilt_sets)
    for splited_set in spilt_sets: 
        for person_folder in sorted(dataset_dir.glob('*')):
            if splited_set in person_folder.name:
                
                output_img_dir = os.path.join(output_dir, splited_set, "images")
                output_xml_dir = os.path.join(output_dir, splited_set, "annotation_xml")
                
                if not os.path.exists(output_img_dir):
                    os.makedirs(output_img_dir)
                if not os.path.exists(output_xml_dir):
                    os.makedirs(output_xml_dir)
                
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
                        
                            video2frames_xml(args, splited_set, video_folder, output_img_dir, output_xml_dir, person_number, video_number, camera)   
                        
                            
                                 


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d','--datatype', type=str, required=True)
    parser.add_argument('-d', '--datatype', type=str, choices=['train', 'val'], required=True, nargs='+', help="Datatype must be 'train' or 'val'")
    # parser.add_argument('-s','--s_device', type=str)
    parser.add_argument('-n','--no_images', type=bool)
    args = parser.parse_args()
    # datatype = args.datatype 
    # s_device = args.s_device
    # print(datatype)
    # print(s_device)
    # if s_device == None:
    #     s_device = "all"


    # cmd
    # python data2voc_GC.py  --datatype train  --s_device tablet 

     
    # abolute dir
    dataset_dir = f'/home/owenserver/storage/Datasets/EVE/eve_dataset'
    output_dir = f'/home/owenserver/storage/Datasets/VOC_format_EVE_Data_ld_ext'
    # dataset_dir = f'/home/owenserver/storage/Datasets/EVE/eve_dataset'
    # output_dir = f'/home/owenserver/storage/Datasets/VOC_format_EVE_Data_ld_3_webcam'
    # dataset_dir = f'/home/owenserver/storage/Datasets/EVE/test_s_eve_dataset'
    # output_dir = f'/home/owenserver/storage/Datasets/EVE/VOC_s_eve_dataset'
    # output_dir = f'/home/owenserver/Python/Gaze_DataProcessing/EVE/VOC_format_EVE_Data_ld_ext'



    create_directory_if_not_exists(output_dir)
    dataset_dir = pathlib.Path(dataset_dir)
    

    ImageProcessing_EVE(dataset_dir,output_dir,args)
    
        


if __name__ == '__main__':
    main()
    