
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

import json
import cv2 

Orientation_counter = np.zeros(4)
GazeCapture_info = []
person_specific_device = []


        
def get_device_info( device_dir: pathlib.Path) -> pd.DataFrame:
    anno_path = device_dir
    select_cols = [0,1,2,3,4,8,11]
    select_names = ['DeviceName','DeviceCameraToScreenXMm', 'DeviceCameraToScreenYMm', 'DeviceCameraXMm', 'DeviceCameraYMm','DeviceScreenWidthMm','DeviceScreenHeightMm']
    df = pd.read_csv(anno_path,
                     delimiter=',',
                     header=None,
                     usecols=select_cols,
                     names=select_names)
    
    # df['day'] = df.path.apply(lambda path: path.split('/')[0])
    # df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    # df = df.drop(['path'], axis=1)
    return df, select_names



def get_image_resolution(image_path):
    try:
        # 開啟影像檔案
        img = Image.open(image_path)

        # 取得影像大小（寬度 x 高度）
        width, height = img.size

        return width, height

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
        
def loadAppleDeviceData(device_csv_dir):
    
    df,names_list = get_device_info(device_csv_dir)

    device_info = np.empty(shape=(0,)) 

    for _, row in df.iterrows():
        dict = {}
        for key in names_list:
            dict[key] = ''
            
        dict['DeviceName'] = row.DeviceName
        dict['DeviceCameraToScreenXMm'] = row.DeviceCameraToScreenXMm
        dict['DeviceCameraToScreenYMm'] = row.DeviceCameraToScreenYMm
        dict['DeviceCameraXMm'] = row.DeviceCameraXMm
        dict['DeviceCameraYMm'] = row.DeviceCameraYMm
        dict['DeviceScreenWidthMm'] = row.DeviceScreenWidthMm
        dict['DeviceScreenHeightMm'] = row.DeviceScreenHeightMm
        
        device_info = np.append(device_info,np.array(dict))
    
    return device_info


def CropImg_WithBlack(img, X, Y, W, H , expand_ratio = 1.2 ):
    
    img_black = np.zeros_like(img)
    Y_lim, X_lim, _ = img.shape
    C_x = int(X + (W/2)) 
    C_y = int(Y + (H/2)) 
    
    H = min(H*expand_ratio, Y_lim)
    W = min(W*expand_ratio, X_lim)
    X = int(C_x - (W/2))
    Y = int(C_y - (H/2))

    X, Y, W, H = list(map(int, [X, Y, W, H]))
    X = max(X, 0)
    Y = max(Y, 0)

    if X + W > X_lim:
        X = X_lim - W

    if Y + H > Y_lim:
        Y = Y_lim - H
        

    img_black[Y:(Y+H),X:(X+W),:] = img[Y:(Y+H),X:(X+W),:]
        
    return img_black

def ImageProcessing_GazeCapture(dataset_dir, device_info, datatype, specify_device):
    persons = os.listdir(dataset_dir)
    persons.sort()
    
    specify_device_list = []
    if specify_device == "phone":
        prefix = "iPhone"
        for index in range(len(device_info)) : 
            deviceName = device_info[index]['DeviceName']
            if deviceName.startswith(prefix): 
                specify_device_list.append(deviceName)     
    elif specify_device == "tablet":
        prefix = "iPad"
        for index in range(len(device_info)) : 
            deviceName = device_info[index]['DeviceName']
            if deviceName.startswith(prefix): 
                specify_device_list.append(deviceName)
    else:
        for index in range(len(device_info)) : 
            deviceName = device_info[index]['DeviceName']
            specify_device_list.append(deviceName)
    
 

    train_count = 0
    for count, person in enumerate(tqdm.tqdm(persons)):
        
        person_dir = os.path.join(dataset_dir, person)
        person_info = json.load(open(os.path.join(person_dir, "info.json")))
        
        

        splited_set = person_info["Dataset"]
        devices = person_info["DeviceName"]
        
        
        # if not os.path.exists(os.path.join(output_dir, splited_set)):
        #     os.makedirs(os.path.join(output_dir,splited_set))
        # if not os.path.exists(os.path.join(output_dir, splited_set, "annotation_xml")):
        #     os.makedirs(os.path.join(output_dir, splited_set, "annotation_xml"))
        # if not os.path.exists(os.path.join(output_dir, splited_set, "image")):
        #     os.makedirs(os.path.join(output_dir, splited_set, "image"))
            
        # output_img_dir = os.path.join(output_dir, splited_set, "image")
        # output_xml_dir = os.path.join(output_dir, splited_set, "annotation_xml")
        
        ######
        # if splited_set == "train" : 
        #     train_count = train_count + 1
        #     if train_count > 50 :
        #         break
        ######
        
        for index in range(len(device_info)) : 
            if index == 0:
                continue
            if device_info[index]['DeviceName'] == devices: 
                person_device_info = device_info[index]
                break
        
        # print(datatype)
        

        if splited_set == datatype  or datatype == None :
            if devices in specify_device_list or devices == None :
                print(devices)
                print(person)
                
                person_specific_device.append(f"{person}")
                
                ImageProcessing_Person(person_dir, person, person_device_info)
        
def ImageProcessing_Person(person_dir, person, person_device_info):
    # Read annotation files
    images_list = json.load(open(os.path.join(person_dir, "frames.json")))
    face_located = json.load(open(os.path.join(person_dir, "appleFace.json")))
    left_located = json.load(open(os.path.join(person_dir, "appleLeftEye.json")))
    right_located = json.load(open(os.path.join(person_dir, "appleRightEye.json")))
    grid_info = json.load(open(os.path.join(person_dir, "faceGrid.json")))
    gt_info = json.load(open(os.path.join(person_dir, "dotInfo.json")))
    screen_info = json.load(open(os.path.join(person_dir, "screen.json")))
    
    skip_repeat = 1
    last_DotNum = None
    
    global Orientation_counter

    for index, frame in enumerate(images_list):
        if not face_located["IsValid"][index]: continue
        if not left_located["IsValid"][index]: continue
        if not right_located["IsValid"][index]: continue
        if not grid_info["IsValid"][index]: continue
        
        if skip_repeat == 1: 
            if gt_info["DotNum"][index] == last_DotNum :
                continue
            last_DotNum = gt_info["DotNum"][index]

        # image_path = os.path.join(person_dir, "frames", frame)
        # print(frame)
        # print(type(person))
        
        
        x1 = int(face_located["X"][index])
        y1 = int(face_located["Y"][index])
        x2 = int(face_located["X"][index] + face_located["W"][index])
        y2 = int(face_located["Y"][index] + face_located["H"][index])
        center_x = int(x1 + (face_located["W"][index]/2)) 
        center_y = int(y1 + (face_located["H"][index]/2)) 
    

        GazeCapture_info.append(f"{x1} {y1} {x2} {y2} {center_x} {center_y}")




def main():
    
    
    output_anno_bbox_txt_path = f'GazeCapture_anno_bbox.txt'
    output_person_specific_device_txt_path = f'GazeCapture_person_specific.txt'
    
    

    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--output_anno_bbox_txt_path', type=str, required=True)
    parser.add_argument('-p','--output_person_specific_device_txt_path', type=str, required=True)
    parser.add_argument('-d','--datatype', type=str, required=True)
    parser.add_argument('-s','--s_device', type=str)
    args = parser.parse_args()
    datatype = args.datatype 
    s_device = args.s_device
    print(datatype)
    print(s_device)
    if s_device == None:
        s_device = "all"
        
    output_anno_bbox_txt_path = args.output_anno_bbox_txt_path 
    output_person_specific_device_txt_path = args.output_person_specific_device_txt_path 
    
    
    output_anno_bbox_txt_base, output_anno_bbox_txt_ext = os.path.splitext(output_anno_bbox_txt_path)
    output_anno_bbox_txt_path = f"{output_anno_bbox_txt_base}_{s_device}_{datatype}{output_anno_bbox_txt_ext}"
    
    
    output_person_specific_device_txt_base, output_person_specific_device_txt_ext = os.path.splitext(output_person_specific_device_txt_path)
    output_person_specific_device_txt_path = f"{output_person_specific_device_txt_base}_{s_device}_{datatype}{output_person_specific_device_txt_ext}"
    

    # cmd
    # python data2voc_GC.py  --datatype train  --s_device tablet 
     
    # dataset_dir = f'/home/owenserver/storage/Datasets/GazeCapture_test'
    # output_dir = f'/home/owenserver/storage/Datasets/GazeCapture_testoutput' 
     
    # abolute dir
    dataset_dir = f'/home/owenserver/storage/Datasets/GazeCapture/Data'
    
    dataset_dir = pathlib.Path(dataset_dir)
    

    #### loadAppleDeviceData ####
    device_csv_dir = "/home/owenserver/storage/Datasets/GazeCapture/apple_device_data.csv"
    device_info = loadAppleDeviceData(device_csv_dir)
    
    # datatype_list = ['train','test']
    # datatype_list = ['test']
    
    datatype_list = []
    datatype_list.append(datatype)
    
    for datatype in datatype_list :
        ImageProcessing_GazeCapture(dataset_dir, device_info, datatype, s_device)
    
    print(Orientation_counter) 
    
    
    
    #### output_txt_path  
    with open(output_anno_bbox_txt_path, 'w', encoding='utf-8') as output_file:
        for info in GazeCapture_info:
            output_file.write(info + '\n')
            
    #### output_person_specific_device_txt_path
    with open(output_person_specific_device_txt_path, 'w', encoding='utf-8') as output_file:
        for info in person_specific_device:
            output_file.write(info + '\n')
        


if __name__ == '__main__':
    main()
    