
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
    select_cols = [0,5,6]
    select_names = ['path', 'x_px', 'y_px',]
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
        

def ImageProcessing_Himax(dataset_dir, output_dir, datatype, s_device, s_setup):
    items = os.listdir(dataset_dir)
    

    folders = [item for item in items if (os.path.isdir(os.path.join(dataset_dir, item)) and s_device in item )]
    print(folders)
    annotations = [item for item in items if (os.path.isfile(os.path.join(dataset_dir, item)) )]

    
    for annotation in annotations:
        # print(annotation)
        if s_device in annotation:
            # print(annotation)
            annotation_path = os.path.join(dataset_dir,annotation)
            annotation_path = pathlib.Path(annotation_path)
            # print(annotation_path)
    
    
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
        
            ImageProcessing_Person(person_dir, output_img_dir, output_xml_dir, annotation_path)
        
        
def ImageProcessing_Person(person_dir, output_img_dir, output_xml_dir, annotation_path):
    
    
    
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
    
    
    # setup = person_dir.parts[-2]
    # person = person_dir.parts[-1]
    filename_number = 1
    for _, row in df.iterrows():
        anno_setup = row.setup
        annp_person = row.person
        
        # filename = row.filename
        
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
            shutil.copy(image_path, output_img_path)
            
            filename_number = filename_number+1
            
            
            img_width, img_height, img_depth = get_image_resolution(image_path)
            
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
            
            
            if  not unit_mm :
                unit_mm.append(ScreenWidthMm)
                unit_mm.append(ScreenHeightMm)

                unit_pixel.append(ScreenWidthPixel)
                unit_pixel.append(ScreenHeightPixel)
            
            screenSize_dict['unit_mm'] = unit_mm
            screenSize_dict['unit_pixel'] = unit_pixel
            
            
            writer = Writer(output_img_path, img_width, img_height, img_depth)
            
            
            writer.addGaze(row.x_px, row.y_px, gaze_origin, gaze_target\
                , camera_dict, monitorPose_dict, screenSize_dict)
            
            
            # write to file
            writer.save(output_xml_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datatype', type=str, required=True)
    parser.add_argument('-s','--s_device', type=str)
    parser.add_argument('-st','--s_setup', type=str)
    parser.add_argument('-fc','--face_crop', action='store_true' )
    args = parser.parse_args()
    datatype = args.datatype 
    s_device = args.s_device
    s_setup = args.s_setup
    face_crop = args.face_crop
    # print(datatype)
    # print(s_device)
    if s_device == None:
        s_device = "all"


    # cmd
    # python data2voc_GC.py  --datatype train  --s_device tablet 
     
    # dataset_dir = f'/home/owenserver/storage/Datasets/GazeCapture_test'
    # output_dir = f'/home/owenserver/storage/Datasets/GazeCapture_testoutput' 
     
    # abolute dir
    dataset_dir = f'/home/owenserver/storage/Datasets/Himax_Gaze/test'
    if  not face_crop:
        output_dir = f'/home/owenserver/storage/Datasets/VOC_format_Himax_{s_device}_{datatype}'
    else:
        output_dir = f'/home/owenserver/storage/Datasets/VOC_format_Himax_facecrop_{s_device}_{datatype}'
    
    if s_device == "all":
        device_list = ["monitor","laptop"]
    else:
        device_list = [s_device]
    

    create_directory_if_not_exists(output_dir)
    dataset_dir = pathlib.Path(dataset_dir)
    

    print(s_device)
    for device in device_list:
        print(device)
        ImageProcessing_Himax(dataset_dir, output_dir, datatype , device, s_setup)
        
    if face_crop :

        # face crop total folder
        input_dir  = f"/home/owenserver/storage/Datasets/VOC_format_Himax_facecrop_{s_device}_{datatype}/{datatype}/images"
        output_dir = f"/home/owenserver/storage/Datasets/VOC_format_Himax_facecrop_{s_device}_{datatype}/{datatype}/images_facecrop"

        input_dir = pathlib.Path(input_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

        image_files = os.listdir(input_dir)
        skip_list = []

        for image_file in tqdm.tqdm(image_files):
            # 构建输入文件的完整路径
            input_path = os.path.join(input_dir, image_file)
            # print(input_path)
            
            
            img = cv2.imread(input_path)

            img_black = np.zeros_like(img)
            expand_ratio = 1.3

            face_locations  = face_recognition.face_locations(img)
            # print(face_locations)
            
            if not face_locations :
                print(image_file)
                skip_list.append(image_file)
                continue
            Wide = abs(face_locations[0][0] - face_locations[0][2])
            Height = abs(face_locations[0][1] - face_locations[0][3])
            
            C_x = int(min(face_locations[0][0],face_locations[0][2]) + (Wide/2)) 
            C_y = int(min(face_locations[0][1],face_locations[0][3]) + (Height/2)) 
            
            expand_half_Wide = int(Wide*expand_ratio/2)
            expand_half_Height = int(Height*expand_ratio/2 )


            img_black[C_x - expand_half_Wide : C_x + expand_half_Wide , C_y - expand_half_Height : C_y + expand_half_Height ] = img[C_x - expand_half_Wide : C_x + expand_half_Wide ,  C_y - expand_half_Height : C_y + expand_half_Height]

            img = img_black
            output_filename = str(image_file)
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, img)
        
        with open(f"/home/owenserver/storage/Datasets/VOC_format_Himax_facecrop_{s_device}_{datatype}/{datatype}/skip_list.txt", "w") as file:
            for item in skip_list:
                file.write(item + "\n")
    
    



if __name__ == '__main__':
    main()
    