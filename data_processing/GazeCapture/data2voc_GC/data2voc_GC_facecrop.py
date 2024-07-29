
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

def ImageProcessing_GazeCapture(dataset_dir, output_dir, device_info, datatype, specify_device):
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
        
        
        if not os.path.exists(os.path.join(output_dir, splited_set)):
            os.makedirs(os.path.join(output_dir,splited_set))
        if not os.path.exists(os.path.join(output_dir, splited_set, "annotation_xml")):
            os.makedirs(os.path.join(output_dir, splited_set, "annotation_xml"))
        if not os.path.exists(os.path.join(output_dir, splited_set, "image")):
            os.makedirs(os.path.join(output_dir, splited_set, "image"))
            
        output_img_dir = os.path.join(output_dir, splited_set, "image")
        output_xml_dir = os.path.join(output_dir, splited_set, "annotation_xml")
        
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
                ImageProcessing_Person(person_dir, output_img_dir, output_xml_dir, person, person_device_info)
        
def ImageProcessing_Person(person_dir, output_img_dir, output_xml_dir, person, person_device_info):
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

        image_path = os.path.join(person_dir, "frames", frame)
        # print(frame)
        # print(type(person))
        person_int = int(person)
        
        img_filename = f"{person_int:04d}{frame}"
        xml_filename = f"{person_int:04d}{os.path.splitext(frame)[0]}.xml"
        

        output_img_path = output_img_dir/ Path(img_filename)
        output_xml_path = output_xml_dir/ Path(xml_filename)
        
        img = cv2.imread(image_path)
        
        img_black = CropImg_WithBlack(img, face_located["X"][index], face_located["Y"][index], 
                        face_located["W"][index], face_located["H"][index],1.3)
        
        # print(img_black.shape)
        # print(output_img_path)
        
        os.chdir(output_img_dir) 
        
        cv2.imwrite(img_filename, img_black)
        # shutil.copy(image_path, output_img_path)
        
        img_width, img_height = get_image_resolution(image_path)
        
        
        # XCam = gt_info["XCam"][index]
        # YCam = gt_info["YCam"][index]
        XPts = gt_info["XPts"][index]
        YPts = gt_info["YPts"][index]
        height_point = screen_info["H"][index]
        width_point = screen_info["W"][index]
        Orientation = screen_info["Orientation"][index]
        
        
        # Orientation count
        # if Orientation == 1:
        #     Orientation_counter[0] +=1 

        # elif Orientation == 2:
        #     Orientation_counter[1] +=1 
        # elif Orientation == 3:
        #     Orientation_counter[2] +=1 
        # elif Orientation == 4:
        #     Orientation_counter[3] +=1 
        
        
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
        
        
        rvects = []
        tvecs = []
        
        # create pascal voc writer (image_path, camera_dict, monitorPose_dict, width, height)
        writer = Writer(output_img_path, img_width, img_height)
        
        
        # gaze_origin = [row.gaze_origin_x,row.gaze_origin_y,row.gaze_origin_z]
        # gaze_target = [row.gaze_target_x,row.gaze_target_y,row.gaze_target_z]
        gaze_origin = []
        gaze_target = []
        
        
        # gazepoint on screen transform
        x_px = XPts 
        y_px = YPts 
        

        if Orientation == 1:
            unit_mm.append(float(person_device_info['DeviceScreenWidthMm']))
            unit_mm.append(float(person_device_info['DeviceScreenHeightMm']))
    
            x_cameraToScreen = ((float(person_device_info['DeviceScreenWidthMm']))/2 - float(person_device_info['DeviceCameraToScreenXMm']))
            y_cameraToScreen = ((float(person_device_info['DeviceScreenHeightMm']))/2 + float(person_device_info['DeviceCameraToScreenYMm']))
            
        elif Orientation == 2:
            unit_mm.append(float(person_device_info['DeviceScreenWidthMm']))
            unit_mm.append(float(person_device_info['DeviceScreenHeightMm']))
            
            x_cameraToScreen = -((float(person_device_info['DeviceScreenWidthMm']))/2 - float(person_device_info['DeviceCameraToScreenXMm']))
            y_cameraToScreen = -((float(person_device_info['DeviceScreenHeightMm']))/2 + float(person_device_info['DeviceCameraToScreenYMm']))
        elif Orientation == 3:
            # with home button on the right
            # width & height switch
            unit_mm.append(float(person_device_info['DeviceScreenHeightMm']))
            unit_mm.append(float(person_device_info['DeviceScreenWidthMm']))
            
            x_cameraToScreen =  ((float(person_device_info['DeviceScreenHeightMm']))/2 + float(person_device_info['DeviceCameraToScreenYMm']))
            y_cameraToScreen = -((float(person_device_info['DeviceScreenWidthMm']))/2 - float(person_device_info['DeviceCameraToScreenXMm']))
        elif Orientation == 4:
            # with home button on the left
            # width & height switch
            unit_mm.append(float(person_device_info['DeviceScreenHeightMm']))
            unit_mm.append(float(person_device_info['DeviceScreenWidthMm']))
            
            x_cameraToScreen = -((float(person_device_info['DeviceScreenHeightMm']))/2 + float(person_device_info['DeviceCameraToScreenYMm']))
            y_cameraToScreen =  ((float(person_device_info['DeviceScreenWidthMm']))/2 - float(person_device_info['DeviceCameraToScreenXMm']))

    
        unit_pixel.append(width_point)
        unit_pixel.append(height_point)

        
        screenSize_dict['unit_mm'] = unit_mm
        screenSize_dict['unit_pixel'] = unit_pixel
        
        z_cameraToScreen = 0
        tvecs.append(float(x_cameraToScreen))
        tvecs.append(float(y_cameraToScreen))
        tvecs.append(float(z_cameraToScreen))
        
        monitorPose_dict['tvecs'] = tvecs
    
        
        writer.addGaze(int(x_px), int(y_px) , gaze_origin, gaze_target\
            , camera_dict, monitorPose_dict, screenSize_dict)

        
        # write to file
        writer.save(output_xml_path)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datatype', type=str, required=True)
    parser.add_argument('-s','--s_device', type=str)
    args = parser.parse_args()
    datatype = args.datatype 
    s_device = args.s_device
    print(datatype)
    print(s_device)
    if s_device == None:
        s_device = "all"

    # cmd
    # python data2voc_GC.py  --datatype train  --s_device tablet 
     
    # dataset_dir = f'/home/owenserver/storage/Datasets/GazeCapture_test'
    # output_dir = f'/home/owenserver/storage/Datasets/GazeCapture_testoutput' 
     
    # abolute dir
    dataset_dir = f'/home/owenserver/storage/Datasets/GazeCapture/Data'
    output_dir = f'/home/owenserver/storage/Datasets/VOC_format_GazeCapture_{s_device}_{datatype}_facecrop_W_13'
    

    create_directory_if_not_exists(output_dir)
    dataset_dir = pathlib.Path(dataset_dir)
    

    #### loadAppleDeviceData ####
    device_csv_dir = "/home/owenserver/storage/Datasets/GazeCapture/apple_device_data.csv"
    device_info = loadAppleDeviceData(device_csv_dir)
    
  
    ImageProcessing_GazeCapture(dataset_dir, output_dir, device_info, datatype, s_device)
    
    print(Orientation_counter) 
        


if __name__ == '__main__':
    main()
    