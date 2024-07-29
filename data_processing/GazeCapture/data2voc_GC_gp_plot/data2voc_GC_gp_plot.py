
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

import matplotlib.pyplot as plt

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

def ImageProcessing_GazeCapture(dataset_dir, device_info, datatype, specify_device , args):
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
                ImageProcessing_Person(person_dir, person, person_device_info, args)
        
def ImageProcessing_Person(person_dir, person, person_device_info, args):
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
    
    screen_pos = [0,0,0,0]

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
        
        
        # gaze_origin = [row.gaze_origin_x,row.gaze_origin_y,row.gaze_origin_z]
        # gaze_target = [row.gaze_target_x,row.gaze_target_y,row.gaze_target_z]
        gaze_origin = []
        gaze_target = []
        
        
        # gazepoint on screen transform
        x_px = 0 
        y_px = 0

        if Orientation == 1:
            x_px = XPts 
            y_px = YPts 
            unit_mm.append(float(person_device_info['DeviceScreenWidthMm']))
            unit_mm.append(float(person_device_info['DeviceScreenHeightMm']))
    
            x_cameraToScreen = ((float(person_device_info['DeviceScreenWidthMm']))/2 - float(person_device_info['DeviceCameraToScreenXMm']))
            y_cameraToScreen = ((float(person_device_info['DeviceScreenHeightMm']))/2 + float(person_device_info['DeviceCameraToScreenYMm']))
            
            
        elif Orientation == 2:
            x_px = XPts
            y_px = YPts
            unit_mm.append(float(person_device_info['DeviceScreenWidthMm']))
            unit_mm.append(float(person_device_info['DeviceScreenHeightMm']))
            
            x_cameraToScreen = -((float(person_device_info['DeviceScreenWidthMm']))/2 - float(person_device_info['DeviceCameraToScreenXMm']))
            y_cameraToScreen = -((float(person_device_info['DeviceScreenHeightMm']))/2 + float(person_device_info['DeviceCameraToScreenYMm']))
        elif Orientation == 3:
            # with home button on the right
            # x_px = YPts  
            # y_px = height_point - XPts 
            x_px = XPts 
            y_px = YPts
            # width & height switch
            unit_mm.append(float(person_device_info['DeviceScreenHeightMm']))
            unit_mm.append(float(person_device_info['DeviceScreenWidthMm']))

            x_cameraToScreen =  ((float(person_device_info['DeviceScreenHeightMm']))/2 + float(person_device_info['DeviceCameraToScreenYMm']))
            y_cameraToScreen = -((float(person_device_info['DeviceScreenWidthMm']))/2 - float(person_device_info['DeviceCameraToScreenXMm']))
        elif Orientation == 4:
            # with home button on the left
            # x_px = width_point - YPts 
            # y_px = XPts 
            x_px = XPts 
            y_px = YPts
            # width & height switch
            unit_mm.append(float(person_device_info['DeviceScreenHeightMm']))
            unit_mm.append(float(person_device_info['DeviceScreenWidthMm']))
            
            x_cameraToScreen = -((float(person_device_info['DeviceScreenHeightMm']))/2 + float(person_device_info['DeviceCameraToScreenYMm']))
            y_cameraToScreen =  ((float(person_device_info['DeviceScreenWidthMm']))/2 - float(person_device_info['DeviceCameraToScreenXMm']))

    
        unit_pixel.append(width_point)
        unit_pixel.append(height_point)

        
        screenSize_dict['unit_mm'] = unit_mm
        screenSize_dict['unit_pixel'] = unit_pixel
        
        
        monitorPose_dict['tvecs'] = tvecs
    
        

        
        # write to file
        # writer.save(output_xml_path)
        
        vp_width, vp_height = 2400, 2400
        vp_pixel_per_mm = 5
        vp = np.zeros((vp_height, vp_width , 1), dtype=np.float32)
        # vp = vp.transpose(1,2,0)
        
        sc_width = int(unit_mm[0]* vp_pixel_per_mm)
        sc_height = int(unit_mm[1]* vp_pixel_per_mm)
        
        # sc_width = width_point
        # sc_height = height_point
        
        print("sc_width sc_height" , sc_width, "  ",sc_height)
        
        if vp_pixel_per_mm > 0 :
            x = int((x_px / width_point) * sc_width)
            y = int((y_px / height_point) * sc_height)
            
        sc_gazepoint = np.array([x,y],dtype=np.int64)
        

        # in mm 
    
        camera_screen_x_offset = x_cameraToScreen * vp_pixel_per_mm
        camera_screen_y_offset = y_cameraToScreen * vp_pixel_per_mm
        
        # camera_screen_x_offset = 0
        # camera_screen_y_offset = 0

          
        vp_gazepoint = [int((vp_width/2)+(sc_gazepoint[0]-(sc_width/2)) + camera_screen_x_offset) ,int((vp_height/2)+(sc_gazepoint[1]-(sc_height/2)) + camera_screen_y_offset)]
        vp_screen = [int((vp_width/2)-(sc_width/2) + camera_screen_x_offset) , int((vp_width/2)+(sc_width/2)+ camera_screen_x_offset),
                     int((vp_height/2)-(sc_height/2) + camera_screen_y_offset), int((vp_height/2)+(sc_height/2)+camera_screen_y_offset)]
        print("vp_screen" , vp_screen)
        # print("shape of vp",vp.shape )
        print("gazepoint in sp" , x_px, "  ",y_px)
        print("cameraToScreen" , x_cameraToScreen, "  ",y_cameraToScreen)
        print("camera_screen_offset" , camera_screen_x_offset, "  ",camera_screen_y_offset)
        print("Orientation" , Orientation)
        print("vp_gazepoint",vp_gazepoint )
        # sel_Orientation = [1,2,3,4]
        
        sel_Orientation = args.orientation
        
        print("sel_Orientation = ", sel_Orientation)
        # sel_Orientation = [2]
        if(Orientation in sel_Orientation):
            
            # if (vp_gazepoint[0] < vp_screen[0] or vp_gazepoint[0] > vp_screen[1]) \
            #     or (vp_gazepoint[1] < vp_screen[2] or vp_gazepoint[1] > vp_screen[3]):
            
                vp[vp_screen[2]:vp_screen[3],vp_screen[0]:vp_screen[1],0] = 0.5
                vp[vp_gazepoint[1]-10:vp_gazepoint[1]+10,vp_gazepoint[0]-10:vp_gazepoint[0]+10, 0] = 1
                
                vp[int(vp_height/2-10):int(vp_height/2+10),int(vp_width/2-10):int(vp_width/2+10), 0] = 1
                # vp[10:30,50:60, 0] = 1

                
                plt.imshow(vp, cmap='gray')  # 使用灰度颜色映射
                plt.axis('off')  # 关闭坐标轴
                
                plt.show()






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datatype', type=str, required=True)
    parser.add_argument('-s','--s_device', type=str)
    
    parser.add_argument('-o', '--orientation', type=int, choices=[1, 2, 3, 4], required=True, nargs='+', help='Orientation choices: 1, 2, 3, or 4')
    args = parser.parse_args()
    datatype = args.datatype 
    s_device = args.s_device
    print(datatype)
    print(s_device)
    if s_device == None:
        s_device = "all"

     
    # abolute dir
    dataset_dir = f'/home/owenserver/storage/Datasets/GazeCapture/Data'
    # output_dir = f'/home/owenserver/storage/Datasets/VOC_format_GazeCapture_plot_{s_device}_{datatype}'
    

    
    dataset_dir = pathlib.Path(dataset_dir)
    

    #### loadAppleDeviceData ####
    device_csv_dir = "/home/owenserver/storage/Datasets/GazeCapture/apple_device_data.csv"
    device_info = loadAppleDeviceData(device_csv_dir)
    
  
    ImageProcessing_GazeCapture(dataset_dir, device_info, datatype, s_device, args)
    
    print(Orientation_counter) 
        


if __name__ == '__main__':
    main()
    