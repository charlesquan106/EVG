
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
        
        
def video2frames_xml(splited_set, video_folder, output_img_dir, output_xml_dir, person_number, video_number, camera):

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
            cmd = f"ffmpeg -i {files_path} {params} {output_image_dir}/{img_filename}"
            # print("cmd: ", cmd)
            subprocess.run(cmd,shell= True, stderr=subprocess.PIPE,stdout=subprocess.DEVNULL)
            
    ##### h5 #####    
    for files_path in sorted(video_folder.glob('*')):        
        if camera_h5 in files_path.name:
            
            h5f = h5py.File(files_path, 'r')
            
            PoG_data = h5f['face_PoG_tobii/data'] 
            PoG_validity = h5f['face_PoG_tobii/validity']
            
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
                if not PoG_validity[index]:
                    print("remove no validity image ", output_image_path.name)
                    os.remove(output_image_path)
                    continue  
                x_px,y_px = PoG_data[index]
                
                
                # anno
                gaze_origin = []
                gaze_target = []
                
                
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
                
                # 1920 x 1080  (25 inch) / 553 x 331 (mm)
                ScreenHeightMm = 331
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
                writer = Writer(output_image_path, img_width, img_height)
                
                
                # gaze_origin = [row.gaze_origin_x,row.gaze_origin_y,row.gaze_origin_z]
                # gaze_target = [row.gaze_target_x,row.gaze_target_y,row.gaze_target_z]
                gaze_origin = []
                gaze_target = []
            
                writer.addGaze(int(x_px), int(y_px) , gaze_origin, gaze_target\
                    , camera_dict, monitorPose_dict, screenSize_dict)
                
                # write to file
                writer.save(output_xml_path)
                
                

            

        
                
                
        

def ImageProcessing_EVE(dataset_dir, output_dir):
    
    
    spilt_sets = ['train','val']
    for splited_set in spilt_sets: 
        for person_folder in sorted(dataset_dir.glob('*')):
            if splited_set in person_folder.name:
                
                output_img_dir = os.path.join(output_dir, splited_set, "image")
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
                        
                        cameras_list = [
                            "webcam_c"]
                        
                        for camera in cameras_list:
                        
                            video2frames_xml(splited_set, video_folder, output_img_dir, output_xml_dir, person_number, video_number, camera)   
                        
                            
                                 


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d','--datatype', type=str, required=True)
    # parser.add_argument('-s','--s_device', type=str)
    # args = parser.parse_args()
    # datatype = args.datatype 
    # s_device = args.s_device
    # print(datatype)
    # print(s_device)
    # if s_device == None:
    #     s_device = "all"

    # cmd
    # python data2voc_GC.py  --datatype train  --s_device tablet 
     
    # dataset_dir = f'/home/owenserver/storage/Datasets/GazeCapture_test'
    # output_dir = f'/home/owenserver/storage/Datasets/GazeCapture_testoutput' 
     
    # abolute dir
    dataset_dir = f'/home/owenserver/storage/Datasets/EVE/test_eve_dataset'
    output_dir = f'/home/owenserver/storage/Datasets/VOC_format_EVE_Data_test'
    

    create_directory_if_not_exists(output_dir)
    dataset_dir = pathlib.Path(dataset_dir)
    


    
  
    ImageProcessing_EVE(dataset_dir,output_dir)
    
        


if __name__ == '__main__':
    main()
    