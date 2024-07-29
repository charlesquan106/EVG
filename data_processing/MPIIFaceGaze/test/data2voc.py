
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



class save_data():
    def __init__(self,gaze_arr,image_arr,pose_arr ):
        self.gaze = gaze_arr
        self.image = image_arr
        self.pose = pose_arr


def get_anno_info(person_id: str, anno_dir: pathlib.Path) -> pd.DataFrame:
    anno_path = anno_dir
    select_cols = [0,1,2,21,22,23,24,25,26]
    select_names = ['path', 'x_px', 'y_px',
                    'gaze_origin_x','gaze_origin_y','gaze_origin_z',
                    'gaze_target_x','gaze_target_y','gaze_target_z',]
    df = pd.read_csv(anno_path,
                     delimiter=' ',
                     header=None,
                     usecols=select_cols,
                     names=select_names)
    
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df


def get_anno_info_landmark(person_id: str, anno_dir: pathlib.Path) -> pd.DataFrame:
    anno_path = anno_dir

    select_cols = [0,1,2, 3,4,5,6,7, 8,9,10,11,12]
    select_names = ['path', 'x_px', 'y_px',
                    'LM_1','LM_2','LM_3','LM_4','LM_5',
                    'LM_6','LM_7','LM_8','LM_9','LM_10',]
    df = pd.read_csv(anno_path,
                     delimiter=' ',
                     header=None,
                     usecols=select_cols,
                     names=select_names)
    
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df

def extract_numbers(string):
    numbers = ""
    is_number_started = False  # 標記是否已經開始提取數字

    for char in string:
        if char.isdigit():
            numbers += char
            is_number_started = True
        elif is_number_started:
            # 如果已經開始提取數字，但當前字符不是數字，則跳出循環
            break

    return int(numbers) if numbers else None



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
    

def copy_files_with_new_names(person_id,day,source_dir, destination_dir):
    # 获取源目录下的所有文件
    files = os.listdir(source_dir)
    person_id_number = extract_numbers(person_id)
    day_number = extract_numbers(day)
    # print(person_id)
    # print(day)    

    
    for filename in files:
        # 构建源文件的完整路径
        source_file = os.path.join(source_dir, filename)
        # 检查是否为文件
        if os.path.isfile(source_file):
            # 构建新的文件名
            base_name, extension = os.path.splitext(filename)
            # print(type(base_name))
            
            base_name_number = extract_numbers(base_name)
            # new_filename = f"{base_name:02d}{extension}"
            new_filename = f"{person_id_number:02d}{day_number:02d}{base_name_number:04d}{extension}"
            
            # 构建目标文件的完整路径
            destination_file = os.path.join(destination_dir, new_filename)
            
            # 复制文件
            shutil.copy(source_file, destination_file)
            # print(f"copy file：{source_file} -> {destination_file}")


def save_one_person(person_id: str, data_dir: pathlib.Path,
                    anno_dir: pathlib.Path, output_xml_dir: pathlib.Path, output_img_dir:pathlib.Path,
                    copy_file, copy_img) -> None:
    filenames = dict()
    data_person_dir = data_dir / person_id
    
    if copy_file == False :
        xml_person_dir = output_xml_dir / person_id
        create_directory_if_not_exists(xml_person_dir)
    
    
    
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
    
    for path in sorted(data_person_dir.glob('*')):
        
        person_id_file = person_id+'.txt'
        day = path.stem
        
        # calibration folder
        # format: .mat
        if(path.name == 'Calibration' ):
            for calib_path in sorted(path.glob('*')):
                
                mat_data = scipy.io.loadmat(calib_path.as_posix(),
                            struct_as_record=False,
                            squeeze_me=True)
                if(calib_path.name == 'Camera.mat' ):
                    camera_dict['cameraMatrix'] = mat_data['cameraMatrix']
                    camera_dict['distCoeffs'] = mat_data['distCoeffs']
                if(calib_path.name == 'monitorPose.mat' ):
                    monitorPose_dict['rvects'] = mat_data['rvects']
                    monitorPose_dict['tvecs'] = mat_data['tvecs']
                if(calib_path.name == 'screenSize.mat' ):
                    unit_mm = []
                    unit_pixel = []
                    unit_mm.append(mat_data['width_mm'])
                    unit_mm.append(mat_data['height_mm'])
                    unit_pixel.append(mat_data['width_pixel'])
                    unit_pixel.append(mat_data['height_pixel'])
                    screenSize_dict['unit_mm'] = unit_mm
                    screenSize_dict['unit_pixel'] = unit_pixel

        # image 
        # format: .jpg
        elif (path.name == day ):
            # print(day)

            if copy_file == True :
                copy_files_with_new_names(person_id, day, path, output_img_dir)
            
            
    for path in sorted(data_person_dir.glob('*')):
        # anno file
        # format: .txt
        if (path.name == person_id_file ):
    
            # df = get_anno_info(person_id, path)
            # df = get_anno_info_landmark(person_id, path)
            df = get_anno_info(person_id, path)
            
            for _, row in df.iterrows():
                day = row.day
        
                filename, _ = os.path.splitext(row.filename)
                
                
                person_id_number = extract_numbers(person_id)
                day_number = extract_numbers(day)
                filename_number = extract_numbers(filename)

                                
                #####  annotation 
                if copy_file == True :
                    
                    img_filename = f"{person_id_number:02d}{day_number:02d}{filename_number:04d}.jpg"
                    xml_filename = f"{person_id_number:02d}{day_number:02d}{filename_number:04d}.xml"

                    output_img_path = output_img_dir/ Path(img_filename)
                    output_xml_path = output_xml_dir/ Path(xml_filename)
                    
                    
                    img_width, img_height = get_image_resolution(output_img_path)
                    
                    # print(img_width, img_height)
                else:
                    # img_filename = str(person_id) + "_"+ str(filename) + ".jpg"  
                    # xml_filename = str(person_id) + "_"+ str(filename) + ".xml"
                    img_filename = row.filename
                    xml_filename = f"{day_number:02d}{filename_number:04d}.xml"
                    
                    output_img_path = data_dir/ Path(person_id) / Path(day) / Path(img_filename)
                    output_xml_path = output_xml_dir/ Path(person_id) / Path(xml_filename)
                    
                    create_directory_if_not_exists(output_xml_dir/ Path(person_id))
                    
                
                
                # create pascal voc writer (image_path, width, height)
                
                # create pascal voc writer (image_path, camera_dict, monitorPose_dict, width, height)
                writer = Writer(output_img_path, img_width, img_height)
                
                gaze_origin = [row.gaze_origin_x,row.gaze_origin_y,row.gaze_origin_z]
                gaze_target = [row.gaze_target_x,row.gaze_target_y,row.gaze_target_z]

                # writer.addGazePoint(row.x_px, row.y_px ,row.LM_1, row.LM_2,row.LM_3,row.LM_4,row.LM_5,row.LM_6,row.LM_7,row.LM_8,row.LM_9,row.LM_10)
                writer.addGaze(row.x_px, row.y_px, gaze_origin, gaze_target\
                    , camera_dict, monitorPose_dict, screenSize_dict)
                # writer.addLandmark(row.LM_1, row.LM_2,row.LM_3,row.LM_4,row.LM_5,row.LM_6,row.LM_7,row.LM_8,row.LM_9,row.LM_10)
                
                
                # write to file
                writer.save(output_xml_path)
                

        

def write_logger(filename, content):
    try:
        # 開啟檔案以寫入模式
        with open(filename, 'a') as file:  # 使用 'a' 模式以追加寫入方式打開檔案
            # 寫入內容
            file.write(content + "\n")  # 添加換行符號以區分每個內容
    except IOError:
        print("error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype', type=str, required=True)
    parser.add_argument('--val_id', type=int, required=True)
    args = parser.parse_args()
    datatype = args.datatype 
    val_id = args.val_id
    datatype_str = f"train/test: {datatype}"
    val_id_str = f"val_id: {val_id}"
    
    official_skip_id_list = [2,7,10]
    custom_skip_id_list = []

    if datatype == "train":
        custom_skip_id_list.append(val_id)         
    else:
        for person_id in range(15):
            if person_id not in official_skip_id_list:
                if person_id != val_id:
                    custom_skip_id_list.append(person_id)
    
    
    total_skip_id_list = []
    total_skip_id_list.extend(custom_skip_id_list)
    total_skip_id_list.extend(official_skip_id_list)
    total_skip_id_list.sort()
    total_skip_id_list_str = f"total_skip_id_list: {total_skip_id_list}"
    # print(f"total_skip_id_list: {total_skip_id_list}")
    

    
     
    # abolute dir
    dataset_dir = f'/home/owenserver/storage/Datasets/MPIIFaceGaze/MPIIFaceGaze'
    output_xml_dir = f'/home/owenserver/storage/Datasets/VOC_format_MPIIFaceGaze_{datatype}_p{val_id:02}/annotation_xml'
    output_img_dir = f'/home/owenserver/storage/Datasets/VOC_format_MPIIFaceGaze_{datatype}_p{val_id:02}/image'
    output_logger_dir = f'/home/owenserver/storage/Datasets/VOC_format_MPIIFaceGaze_{datatype}_p{val_id:02}/logger.txt'

    # related dir
    # dataset_dir = 'datasets/MPIIFaceGaze'
    # output_xml_dir = 'datasets/VOC_format_MPIIFaceGaze/annotation'
    # output_img_dir = 'datasets/VOC_format_MPIIFaceGaze/image'

    create_directory_if_not_exists(output_xml_dir)
    create_directory_if_not_exists(output_img_dir)
    
    write_logger(output_logger_dir,datatype_str)
    write_logger(output_logger_dir,val_id_str)
    write_logger(output_logger_dir,total_skip_id_list_str)

    
        
    # copy_file = 1
    # 則會將現有的影像重新編號，並且依照PPDDIIII的格式命名，並且將複製全部影像都放置於output_img_dir 資料夾內，並且會
    # 以新編號的path與filename訊息儲存於annotation內
    
    # copy_file = 0
    # 則會保持原有的影像，不會複製影像至output_img_dir，僅生成對對應的annotation，但要注意的是，會將annotationt儲存於output_xml_dir
    # 並且僅保留person資料夾，而dayxx會被一併的消失，取而代之的是filename 會加註dayxx -> DDIIII，因為發現不同的dayxx 內的編號並不是單一
    # person唯一，僅是dayxx 之下唯一，所以當全部的dayxx 被消失之後，必須要變換為 DDIIII 才不會有重複的問題
    copy_file = 1
    copy_img = 0
    dataset_dir = pathlib.Path(dataset_dir)
    output_xml_dir = pathlib.Path(output_xml_dir)
    output_img_dir = pathlib.Path(output_img_dir)
    

    for person_id in tqdm.tqdm(range(15)):
        if person_id not in total_skip_id_list:
            person_id = f'p{person_id:02}'
            data_dir = dataset_dir 
            anno_dir = dataset_dir
            # print(person_id)
            save_one_person(person_id, data_dir, anno_dir, output_xml_dir, output_img_dir, copy_file, copy_img)
        else: 
            pass
            # print(f"skip: {person_id}")
            
        # person_id = f'p{person_id:02}'
        # data_dir = dataset_dir 
        # anno_dir = dataset_dir
        # print(person_id)
        # save_one_person(person_id, data_dir, anno_dir, output_xml_dir, output_img_dir, copy_file)


if __name__ == '__main__':
    main()
    