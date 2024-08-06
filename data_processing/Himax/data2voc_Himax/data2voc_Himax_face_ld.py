import scipy.io
import numpy as np

import pandas as pd
import pathlib
from pathlib import Path
from PIL import Image
from pascal_voc_writer import Writer

import argparse
import pathlib

import pandas as pd
import tqdm
import os
import shutil
import cv2
import face_recognition

import json

Orientation_counter = np.zeros(4)

        
def get_anno_info(anno_path: pathlib.Path) -> pd.DataFrame:
    
    print("-----------------------")
    select_cols = [0,3,4,5,6,7,8,9]
    select_names = ['path','gaze_target_x','gaze_target_y', 'x_px', 'y_px','face_o_x', 'face_o_y', 'face_o_z',]
    df = pd.read_csv(anno_path,
                     sep=r'[ ,]',
                     header=None,
                     usecols=select_cols,
                     names=select_names,
                     engine='python')
    
    df['setup'] = df.path.apply(lambda path: path.split('/')[2])
    df['person'] = df.path.apply(lambda path: path.split('/')[3])
    df['filename'] = df.path.apply(lambda path: path.split('/')[4])
    df = df.drop(['path'], axis=1)
    return df

def get_anno_info_face_bbox(anno_path: pathlib.Path) -> pd.DataFrame:
    
    print("-----------------------")
    select_cols = [0,1,2,3,4]
    select_names = ['path', 'xmin', 'ymin', 'xmax', 'ymax' ]
    df = pd.read_csv(anno_path,
                     sep=r'[ ,]',
                     header=None,
                     usecols=select_cols,
                     names=select_names,
                     engine='python')
    
    df['setup'] = df.path.apply(lambda path: path.split('/')[2])
    df['person'] = df.path.apply(lambda path: path.split('/')[3])
    df['filename'] = df.path.apply(lambda path: path.split('/')[4])
    df = df.drop(['path'], axis=1)
    return df


def get_image_resolution(image_path):
    try:
        # 開啟影像檔案
        img = Image.open(image_path)

        # 取得影像大小（寬度 x 高度）
        width, height = img.size
        
        # depth = img.mode
        
        channels = img.getbands()
        depth = len(channels)

        # 关闭图像文件
        img.close()

        return width, height, depth

    except IOError:
        print(f"無法開啟影像：{image_path}")
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
    



def write_logger(filename, content):
    try:
        # 開啟檔案以寫入模式
        with open(filename, 'a') as file:  # 使用 'a' 模式以追加寫入方式打開檔案
            # 寫入內容
            file.write(content + "\n")  # 添加換行符號以區分每個內容
    except IOError:
        print("error")
        

def ImageProcessing_Himax(dataset_dir, output_dir, datatype, s_device, no_images):
    items = os.listdir(dataset_dir)
    print("dataset_dir = ", dataset_dir)

    folders = [item for item in items if (os.path.isdir(os.path.join(dataset_dir, item)) and s_device in item )]
    print(folders)
    annotations = [item for item in items if (os.path.isfile(os.path.join(dataset_dir, item)) )]
    
    items_bbox = os.listdir(os.path.join(dataset_dir,'FD_bbox'))
    

    annotations_bbox = [item for item in items_bbox if (os.path.isfile(os.path.join(dataset_dir,'FD_bbox', item)) )]
    # print("annotations_bbox = ", annotations_bbox)
    
    for annotation in annotations:
        # print(annotation)
        if s_device in annotation:
            # print(annotation)
            annotation_path = os.path.join(dataset_dir,annotation)
            annotation_path = pathlib.Path(annotation_path)
    
    
    for annotation_bbox in annotations_bbox:
        # print(annotation)
        if s_device in annotation_bbox:        
            annotation_bbox_path = os.path.join(dataset_dir,'FD_bbox', annotation_bbox)
            annotation_bbox_path = pathlib.Path(annotation_bbox_path)
            
            print("annotation_bbox_path = ",annotation_bbox_path)
    
    
    if not os.path.exists(os.path.join(output_dir, datatype)):
        os.makedirs(os.path.join(output_dir,datatype))
    if not os.path.exists(os.path.join(output_dir, datatype, "annotation_xml")):
        os.makedirs(os.path.join(output_dir, datatype, "annotation_xml"))
    if not os.path.exists(os.path.join(output_dir, datatype, "images")):
        os.makedirs(os.path.join(output_dir, datatype, "images"))
        
    output_img_dir = os.path.join(output_dir, datatype, "images")
    output_xml_dir = os.path.join(output_dir, datatype, "annotation_xml")
    
    
    # for count, folder  in enumerate(tqdm.tqdm(folders)):
    for count, setup  in enumerate(folders):
        setup_path = os.path.join(dataset_dir, setup)
        items = os.listdir(setup_path)
        persons = [item for item in items if os.path.isdir(os.path.join(setup_path, item)) ]
        # print(persons)
        
        # for count, person  in enumerate(tqdm.tqdm(persons)):
        for count, person  in enumerate(persons):
            person_dir = os.path.join(setup_path, person)
            # print(person_dir)
            person_dir = pathlib.Path(person_dir)
        
            ImageProcessing_Person(person_dir, output_img_dir, output_xml_dir, annotation_path, annotation_bbox_path, no_images)
        
        
def ImageProcessing_Person(person_dir, output_img_dir, output_xml_dir, annotation_path, annotation_bbox_path, no_images):
    
    
    
    # device_define = {"laptop":{
    #             "width_px": 1366,
    #             "width_mm": 295,
    #             "height_px": 768,
    #             "height_mm": 166,},
    #         "monitor":{
    #             "width_px": 1920,
    #             "width_mm": 480,
    #             "height_px": 1080,
    #             "height_mm": 270,},}
    
    device_define = {"laptop":{
                "width_px": 1366,
                "width_mm": 295,
                "height_px": 768,
                "height_mm": 166,},
            "monitor":{
                "width_px": 1366,
                "width_mm": 295,
                "height_px": 768,
                "height_mm": 166,},}
    
    unit_mm = []
    unit_pixel = []
    print("Folder : ",  person_dir.parts[-2])
    for device_dict in device_define.items():
        if device_dict[0] == annotation_path.stem:
            ScreenHeightMm      = device_dict[1]["height_mm"]
            ScreenWidthMm       = device_dict[1]["width_mm"]
            ScreenHeightPixel   = device_dict[1]["height_px"]
            ScreenWidthPixel    = device_dict[1]["width_px"]
            
            # print(ScreenHeightMm)
    
    # print(person_dir)
    df = get_anno_info(annotation_path)

    df_face_bbox = get_anno_info_face_bbox(annotation_bbox_path)

    
    # setup = person_dir.parts[-2]
    # person = person_dir.parts[-1]
    filename_number = 1
    for _, row in df.iterrows():
        anno_setup = row.setup
        annp_person = row.person

        
        
        
        if anno_setup == person_dir.parts[-2] and annp_person == person_dir.parts[-1] :
            annp_filename = row.filename
            # print(annp_filename)

            
            img_filename = f"{anno_setup}_{annp_person}_{filename_number:04d}.jpg"
            xml_filename = f"{anno_setup}_{annp_person}_{filename_number:04d}.xml"            
            
            output_img_path = output_img_dir/ Path(img_filename)
            output_xml_path = output_xml_dir/ Path(xml_filename)
            
            image_path = os.path.join(person_dir, annp_filename)
            img = cv2.imread(image_path)
            # print("img.shape: ",img.shape)
            
            
            # print(image_path)
            if no_images != True :
                shutil.copy(image_path, output_img_path)
            
            filename_number = filename_number+1
            
            
            img_width, img_height, img_depth = get_image_resolution(image_path)
            
            gaze_origin = [row.face_o_x, row.face_o_y, row.face_o_z ]
            screen_depth_mm_offset = -15  # measure manually
            gaze_target = [row.gaze_target_x, row.gaze_target_y, screen_depth_mm_offset]
            
            
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
            
            
            if  not unit_mm :
                unit_mm.append(ScreenWidthMm)
                unit_mm.append(ScreenHeightMm)

                unit_pixel.append(ScreenWidthPixel)
                unit_pixel.append(ScreenHeightPixel)
            
            screenSize_dict['unit_mm'] = unit_mm
            screenSize_dict['unit_pixel'] = unit_pixel
            
            
            face_bbox = [] 
            
            df_face_bbox_row = df_face_bbox[df_face_bbox['filename'] == annp_filename]
            # print(df_face_bbox_row)
            # print()
            if not df_face_bbox_row.empty:
                xmin = df_face_bbox_row.iloc[0]['xmin']
                # print(xmin)
                ymin = df_face_bbox_row.iloc[0]['ymin']
                xmax = df_face_bbox_row.iloc[0]['xmax']
                ymax = df_face_bbox_row.iloc[0]['ymax']
                face_bbox = [xmin,ymin,xmax,ymax]
            

            
            
            writer = Writer(output_img_path, img_width, img_height, img_depth)
            
            
            writer.addGaze(row.x_px, row.y_px, gaze_origin, gaze_target\
                , camera_dict, monitorPose_dict, screenSize_dict, face_bbox)
            
            
            # write to file
            writer.save(output_xml_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datatype', type=str, required=True)
    parser.add_argument('-n','--no_images', type=bool)
    parser.add_argument('-s','--s_device', type=str)
    # parser.add_argument('-n','--no_images', type=bool)
    args = parser.parse_args()
    datatype = args.datatype 
    s_device = args.s_device
    no_images = args.no_images
    no_images = True
    print("no_images = ", no_images)

    # print(datatype)
    # print(s_device)
    if s_device == None:
        s_device = "all"


    # cmd
    # python data2voc_GC.py  --datatype train  --s_device tablet 
     
    # dataset_dir = f'/home/owenserver/storage/Datasets/GazeCapture_test'
    # output_dir = f'/home/owenserver/storage/Datasets/GazeCapture_testoutput' 
     
    # abolute dir
    # dataset_dir = f'/home/owenserver/storage/Datasets/Himax_Gaze/rgb_test'
    # output_dir = f'/home/owenserver/storage/Datasets/Himax_Gaze/rgb_test/VOC_format_Himax_s_{s_device}_{datatype}'
    
    dataset_dir = f'/home/owenserver/storage/Datasets/Himax_Gaze/rgb_{datatype}'
    output_dir = f'/home/owenserver/storage/Datasets/VOC_format_Himax_{s_device}_rgb_{datatype}_tttt'

    if s_device == "all":
        device_list = ["monitor","laptop"]
    else:
        device_list = [s_device]
    

    create_directory_if_not_exists(output_dir)
    dataset_dir = pathlib.Path(dataset_dir)
    

    print(s_device)
    for device in device_list:
        print(device)
        ImageProcessing_Himax(dataset_dir, output_dir, datatype , device , no_images)
        



if __name__ == '__main__':
    main()
    