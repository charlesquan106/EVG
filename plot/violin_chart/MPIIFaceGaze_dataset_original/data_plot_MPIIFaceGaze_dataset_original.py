
import scipy.io
import numpy as np

import pandas as pd
import pathlib
from pathlib import Path
from PIL import Image

import argparse
import pathlib

import pandas as pd
import tqdm
import os
import shutil

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import LogNorm



all_dfs = []

class save_data():
    def __init__(self,gaze_arr,image_arr,pose_arr ):
        self.gaze = gaze_arr
        self.image = image_arr
        self.pose = pose_arr


    

def get_anno_info(person_id: str, anno_dir: pathlib.Path) -> pd.DataFrame:
    anno_path = anno_dir
    select_cols = [3,4,5,6,7,8,9,10,11,12,13,14
                   ,15,16,17
                   ,18,19,20
                   ,21,22,23
                   ,24,25,26]
    select_names = [
                    'LM_1','LM_2','LM_3','LM_4','LM_5','LM_6',
                    'LM_7','LM_8','LM_9','LM_10','LM_11','LM_12',
                    'h_R_x', 'h_R_y', 'h_R_z',
                    'h_T_x', 'h_T_y', 'h_T_z',
                    'gaze_origin_x','gaze_origin_y','gaze_origin_z',
                    'gaze_target_x','gaze_target_y','gaze_target_z',]
    df = pd.read_csv(anno_path,
                     delimiter=' ',
                     header=None,
                     usecols=select_cols,
                     names=select_names)
    
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


# 定義函數來計算並標準化視線向量
def calculate_normalized_vector(row):
    origin = np.array([row['gaze_origin_x'], row['gaze_origin_y'], row['gaze_origin_z']])
    target = np.array([row['gaze_target_x'], row['gaze_target_y'], row['gaze_target_z']])
    vector = target - origin
    length = np.linalg.norm(vector)
    if length != 0:
        normalized_vector = vector / length
    else:
        normalized_vector = vector  # 若長度為0，則視線向量不變
    return normalized_vector


def vector_to_angle(vector: np.ndarray) -> np.ndarray:
    assert vector.shape == (3, )
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw])


def calculate_vector_to_angle(row):
    vector = np.array([row['normalized_vector_x'], row['normalized_vector_y'], row['normalized_vector_z']])

    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    angle = np.rad2deg(np.array([pitch, yaw]))
    
    return angle


def calculate_mean_face(row):
    mean_x = int(np.mean(np.array([row['LM_1'], row['LM_3'], row['LM_5'],row['LM_7'], row['LM_9'], row['LM_11']])))
    mean_y = int(np.mean(np.array([row['LM_2'], row['LM_4'], row['LM_6'],row['LM_8'], row['LM_10'], row['LM_12']])))

    mean = np.array([mean_x, mean_y])
    return mean
        

def save_one_person(person_id: str, data_dir: pathlib.Path,
                    anno_dir: pathlib.Path) -> None:
    filenames = dict()
    data_person_dir = data_dir / person_id
    

    
    
    
    
    
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
            print(day)

            # if copy_file == True :
            #     copy_files_with_new_names(person_id, day, path, output_img_dir)
            
            
    for path in sorted(data_person_dir.glob('*')):
        # anno file
        # format: .txt
        if (path.name == person_id_file ):
    
            # df = get_anno_info(person_id, path)
            # df = get_anno_info_landmark(person_id, path)
            df = get_anno_info(person_id, path)
            
            # print(df.head())
            print(df.shape)
            
            df['person_id'] = person_id
            
            
            

            
            df['mean'] = df.apply(calculate_mean_face, axis=1)
            
            df['mean_x'] = df['mean'].apply(lambda vec: vec[0])
            df['mean_y'] = df['mean'].apply(lambda vec: vec[1])
            
            df = df.drop(['mean'], axis=1)
            
            

            # 應用函數到每一行，並新增為一個列 'normalized_vector'
            df['normalized_vector'] = df.apply(calculate_normalized_vector, axis=1)
            
            # 拆分為三個新列
            df['normalized_vector_x'] = df['normalized_vector'].apply(lambda vec: vec[0])
            df['normalized_vector_y'] = df['normalized_vector'].apply(lambda vec: vec[1])
            df['normalized_vector_z'] = df['normalized_vector'].apply(lambda vec: vec[2])
            
            df = df.drop(['normalized_vector'], axis=1)
            
            
            
            
            df['normalized_vector_angle'] = df.apply(calculate_vector_to_angle, axis=1)
            
        
            df['normalized_vector_angle_pitch'] = df['normalized_vector_angle'].apply(lambda vec: vec[0])
            df['normalized_vector_angle_yaw'] = df['normalized_vector_angle'].apply(lambda vec: vec[1])

            df = df.drop(['normalized_vector_angle'], axis=1)
        
            
            
            all_dfs.append(df)
            
            # # 繪製 x_px 的小提琴圖
            # plt.figure(figsize=(12, 8))
            # sns.violinplot(x='day', y='x_px', data=df, inner='quartile', palette='muted')

            # # 添加標題和標籤
            # plt.title('Violin Plot of x_px for Different Days')
            # plt.xlabel('Day')
            # plt.ylabel('x_px')

            # # 顯示圖表
            # plt.show()
            
            # for _, row in df.iterrows():
            #     day = row.day
        
            #     filename, _ = os.path.splitext(row.filename)
                
                
            #     person_id_number = extract_numbers(person_id)
            #     day_number = extract_numbers(day)
            #     filename_number = extract_numbers(filename)

                                
            #     #####  annotation 
            #     if copy_file == True :
                    
            #         img_filename = f"{person_id_number:02d}{day_number:02d}{filename_number:04d}.jpg"
            #         xml_filename = f"{person_id_number:02d}{day_number:02d}{filename_number:04d}.xml"

            #         output_img_path = output_img_dir/ Path(img_filename)
            #         output_xml_path = output_xml_dir/ Path(xml_filename)
                    
                    
                    
            #         # print(img_width, img_height)
            #     else:
            #         # img_filename = str(person_id) + "_"+ str(filename) + ".jpg"  
            #         # xml_filename = str(person_id) + "_"+ str(filename) + ".xml"
            #         img_filename = row.filename
            #         xml_filename = f"{day_number:02d}{filename_number:04d}.xml"
                    
            #         output_img_path = data_dir/ Path(person_id) / Path(day) / Path(img_filename)
            #         output_xml_path = output_xml_dir/ Path(person_id) / Path(xml_filename)
                    
            #         create_directory_if_not_exists(output_xml_dir/ Path(person_id))
                    
                

        

def write_logger(filename, content):
    try:
        # 開啟檔案以寫入模式
        with open(filename, 'a') as file:  # 使用 'a' 模式以追加寫入方式打開檔案
            # 寫入內容
            file.write(content + "\n")  # 添加換行符號以區分每個內容
    except IOError:
        print("error")


def main():

    
    # total_skip_id_list = [1,2,3,4,5,6,7,8,9]
    total_skip_id_list = []
    total_skip_id_list.sort()
    print(f"total_skip_id_list: {total_skip_id_list}")
    

    dataset_dir = f'/home/master_111/nm6114091/Python/test/violin_chart/MPIIFaceGaze_dataset_original/MPIIFaceGaze_data_dataset_original'

    dataset_dir = pathlib.Path(dataset_dir)

    

    for person_id in tqdm.tqdm(range(15)):
        if person_id not in total_skip_id_list:
            person_id = f'p{person_id:02}'
            data_dir = dataset_dir 
            anno_dir = dataset_dir
            print(person_id)
            save_one_person(person_id, data_dir, anno_dir)
        else: 
            pass
            print(f"skip: {person_id}")
            
        
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print("combined_df = ")
    print(combined_df.shape)
    
    print(combined_df.head())
    
    # 繪製 x_px 的小提琴圖
    # plt.figure(figsize=(12, 8))
    # sns.violinplot(x='person_id', y='x_px', data=combined_df, inner='quartile', palette='muted')
    
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    # sns.violinplot(x='person_id', y='h_R_x', data=combined_df)
    sns.violinplot(y='h_R_x', data=combined_df)
    plt.title('head pose R_x')

    plt.subplot(1, 3, 2)
    # sns.violinplot(x='person_id', y='h_R_y', data=combined_df)
    sns.violinplot(y='h_R_y', data=combined_df)
    plt.title('head pose R_y')
    
    plt.subplot(1, 3, 3)
    # sns.violinplot(x='person_id', y='h_R_z', data=combined_df)
    sns.violinplot(y='h_R_z', data=combined_df)
    plt.title('head pose R_z')
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    # sns.violinplot(x='person_id', y='h_T_x', data=combined_df)
    sns.violinplot(y='h_T_x', data=combined_df)
    plt.title('head pose T_x')

    plt.subplot(1, 3, 2)
    # sns.violinplot(x='person_id', y='h_T_y', data=combined_df)
    sns.violinplot(y='h_T_y', data=combined_df)
    plt.title('head pose T_y')
    
    plt.subplot(1, 3, 3)
    # sns.violinplot(x='person_id', y='h_T_z', data=combined_df)
    sns.violinplot(y='h_T_z', data=combined_df)
    plt.title('head pose T_z')
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    # sns.violinplot(x='person_id', y='normalized_vector_x', data=combined_df)
    sns.violinplot( y='normalized_vector_x', data=combined_df)
    plt.title('gaze vector x')

    plt.subplot(1, 3, 2)
    # sns.violinplot(x='person_id', y='normalized_vector_y', data=combined_df)
    sns.violinplot( y='normalized_vector_y', data=combined_df)
    plt.title('gaze vector y')
    
    plt.subplot(1, 3, 3)
    # sns.violinplot(x='person_id', y='normalized_vector_z', data=combined_df)
    sns.violinplot( y='normalized_vector_z', data=combined_df)
    plt.title('gaze vector z')
    
    
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    sns.violinplot(y='normalized_vector_angle_pitch', data=combined_df)
    plt.title('gaze angle_pitch')

    plt.subplot(1, 2, 2)
    # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    sns.violinplot(y='normalized_vector_angle_yaw', data=combined_df)
    plt.title('gaze angle_yaw')
    
    # 顯示圖表
    plt.show()
    
    
    

    
    
    
    # data_x = combined_df['normalized_vector_angle_yaw'].values
    # data_y = combined_df['normalized_vector_angle_pitch'].values
    # data = np.vstack((data_x, data_y)).T

    # fig, ax = plt.subplots(figsize=(6, 6))

    # # Heatmap plot function with detailed grid
    # heatmap, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=np.arange(-20, 22, 1))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # im = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')

    # # Set grid
    # ax.set_xticks(np.arange(-20, 22, 1))  # Grid lines at each unit for x-axis
    # ax.set_yticks(np.arange(-20, 22, 1))  # Grid lines at each unit for y-axis
    # ax.grid(True, color='white', linestyle='-', linewidth=0.5)


    # # # # Set minor ticks for the grid lines
    # # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # # ax.yaxis.set_minor_locator(MultipleLocator(1))



    # # Customize ticks and labels
    # ax.set_xlabel('Yaw [deg]')
    # ax.set_ylabel('Pitch [deg]')
    # ax.set_xlim([-20, 21])
    # ax.set_ylim([-20, 21])
    # ax.set_title('Detailed Grid Heatmap')



    # # Add colorbar
    # plt.colorbar(im, ax=ax)
    # plt.tight_layout()
    # plt.show()
    
    
    data_x = combined_df['normalized_vector_angle_yaw'].values
    data_y = combined_df['normalized_vector_angle_pitch'].values
    x_range = (-80, 80)
    y_range = (-80, 80)
    
    
    # 創建 2D 直方圖
    plt.hist2d(data_x, data_y, bins=160, range=[x_range, y_range], cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Yaw (deg)')
    plt.ylabel('Pitch (deg)')
    plt.title('Gaze Direction')
    plt.show()
    
    
    
    
    
    # data_x = combined_df['mean_x'].values
    # data_y = combined_df['mean_y'].values
    # data = np.vstack((data_x, data_y)).T
    

    # # 绘制热图
    # fig, ax = plt.subplots(figsize=(12, 8))  # 调整 figsize 以适应新的分辨率

    # # Heatmap plot function with detailed grid
    # heatmap, xedges, yedges = np.histogram2d(data_x, data_y, bins=[np.arange(0, 1281, 10), np.arange(0, 721, 10)])
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # im = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')

    # # Set grid
    # ax.set_xticks(np.arange(0, 1281, 100))  # Grid lines at every 100 units for x-axis
    # ax.set_yticks(np.arange(0, 721, 50))   # Grid lines at every 50 units for y-axis
    # ax.grid(True, color='white', linestyle='-', linewidth=0.5)

    # # Customize ticks and labels
    # ax.set_xlabel('x [pixels]')
    # ax.set_ylabel('y [pixels]')
    # ax.set_xlim([0, 1280])
    # ax.set_ylim([0, 720])
    # ax.set_title('Detailed Grid Heatmap')

    # # Add colorbar
    # plt.colorbar(im, ax=ax)
    # plt.tight_layout()
    # plt.show()
    
    
    
    data_x = combined_df['mean_x'].values
    data_y = combined_df['mean_y'].values
    x_range = (0, 1280)
    y_range = (0, 720)
    
    
    # 創建 2D 直方圖
    plt.hist2d(data_x, data_y, bins=100, range=[x_range, y_range], cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.title('Head Position in Image')
    plt.show()


if __name__ == '__main__':
    main()
    