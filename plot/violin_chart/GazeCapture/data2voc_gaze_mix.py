
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



class save_data():
    def __init__(self,gaze_arr,image_arr,pose_arr ):
        self.gaze = gaze_arr
        self.image = image_arr
        self.pose = pose_arr


    

def get_anno_info( anno_dir: pathlib.Path) -> pd.DataFrame:
    anno_path = anno_dir
    select_cols = [2,3,
                   4,5,]
    select_names = [
                    'normalized_h_rad_pitch','normalized_h_rad_yaw',
                    'normalized_vector_rad_pitch','normalized_vector_rad_yaw',]
    df = pd.read_csv(anno_path,
                     delimiter=' ',
                     header=None,
                     usecols=select_cols,
                     names=select_names)
    
    return df

# def get_anno_info( anno_dir: pathlib.Path) -> pd.DataFrame:
#     anno_path = anno_dir
#     select_cols = [
#                    6,7,
#                    8,9,
#                    ]
#     select_names = [

#                     'normalized_vector_rad_pitch','normalized_vector_rad_yaw',
#                     'normalized_h_rad_pitch','normalized_h_rad_yaw',
# ]
#     df = pd.read_csv(anno_path,
#                      delimiter=' ',
#                      header=None,
#                      usecols=select_cols,
#                      names=select_names)
    
#     return df



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


def calculate_rad_to_angle_vector(row):


    angle = np.rad2deg(np.array([row['normalized_vector_rad_pitch'], row['normalized_vector_rad_yaw']]))
    
    return angle

def calculate_rad_to_angle_h(row):


    angle = np.rad2deg(np.array([row['normalized_h_rad_pitch'], row['normalized_h_rad_yaw']]))
    
    return angle



def calculate_mean_face(row):
    mean_x = int(np.mean(np.array([row['LM_1'], row['LM_3'], row['LM_5'],row['LM_7'], row['LM_9'], row['LM_11']])))
    mean_y = int(np.mean(np.array([row['LM_2'], row['LM_4'], row['LM_6'],row['LM_8'], row['LM_10'], row['LM_12']])))

    mean = np.array([mean_x, mean_y])
    return mean
        

def data_df( data_dir: pathlib.Path):
    
    
            all_dfs = []
    
            df = get_anno_info(data_dir)
            
            # print(df.head())
            print(df.shape)
            
            
        
            
            
            
            # df['normalized_vector_angle'] = df.apply(calculate_vector_to_angle, axis=1)
            
        
            # df['normalized_vector_angle_pitch'] = df['normalized_vector_angle'].apply(lambda vec: vec[0])
            # df['normalized_vector_angle_yaw'] = df['normalized_vector_angle'].apply(lambda vec: vec[1])

            # df = df.drop(['normalized_vector_angle'], axis=1)
        
            print(df.head())
            
            
            df['normalized_vector_angle'] = df.apply(calculate_rad_to_angle_vector, axis=1)
            
            df['normalized_vector_angle_pitch'] = df['normalized_vector_angle'].apply(lambda vec: vec[0])
            df['normalized_vector_angle_yaw'] = df['normalized_vector_angle'].apply(lambda vec: vec[1])

            df = df.drop(['normalized_vector_angle'], axis=1)
            
            df = df.drop(['normalized_vector_rad_pitch'], axis=1)
            df = df.drop(['normalized_vector_rad_yaw'], axis=1)
            
            df['normalized_h_angle'] = df.apply(calculate_rad_to_angle_h, axis=1)
            
            df['normalized_h_angle_pitch'] = df['normalized_h_angle'].apply(lambda vec: vec[0])
            df['normalized_h_angle_yaw'] = df['normalized_h_angle'].apply(lambda vec: vec[1])

            df = df.drop(['normalized_h_angle'], axis=1)
            
            df = df.drop(['normalized_h_rad_pitch'], axis=1)
            df = df.drop(['normalized_h_rad_yaw'], axis=1)
            
            all_dfs.append(df)
            
            
            return all_dfs



        

def write_logger(filename, content):
    try:
        # 開啟檔案以寫入模式
        with open(filename, 'a') as file:  # 使用 'a' 模式以追加寫入方式打開檔案
            # 寫入內容
            file.write(content + "\n")  # 添加換行符號以區分每個內容
    except IOError:
        print("error")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--datadir', type=str, required=True)
    # args = parser.parse_args()
    # datadir = args.datadir 
    
    
    
    datadir_1 = '/home/master_111/nm6114091/Python/test/violin_chart/GazeCapture/GazeCapture_data/GazeCapture_anno_gaze_train.txt'
    datadir_2 = '/home/master_111/nm6114091/Python/test/violin_chart/GazeCapture/GazeCapture_data/GazeCapture_anno_gaze_train_phone.txt'
    datadir_3 = '/home/master_111/nm6114091/Python/test/violin_chart/GazeCapture/GazeCapture_data/GazeCapture_anno_gaze_train_tablet.txt'
    
    all_dfs_1 =  data_df(datadir_1)
    all_dfs_2 =  data_df(datadir_2)
    all_dfs_3 =  data_df(datadir_3)
  
    combined_df_1 = pd.concat(all_dfs_1, ignore_index=True)
    combined_df_2 = pd.concat(all_dfs_2, ignore_index=True)
    combined_df_3 = pd.concat(all_dfs_3, ignore_index=True)
        
 

    

    
    # 繪製 x_px 的小提琴圖
    # plt.figure(figsize=(12, 8))
    # sns.violinplot(x='person_id', y='x_px', data=combined_df, inner='quartile', palette='muted')
    
    
    
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 3, 1)
    # # sns.violinplot(x='person_id', y='h_R_x', data=combined_df)
    # sns.violinplot(y='h_R_x', data=combined_df)
    # plt.title('head pose R_x')

    # plt.subplot(1, 3, 2)
    # # sns.violinplot(x='person_id', y='h_R_y', data=combined_df)
    # sns.violinplot(y='h_R_y', data=combined_df)
    # plt.title('head pose R_y')
    
    # plt.subplot(1, 3, 3)
    # # sns.violinplot(x='person_id', y='h_R_z', data=combined_df)
    # sns.violinplot(y='h_R_z', data=combined_df)
    # plt.title('head pose R_z')
    
    
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 3, 1)
    # # sns.violinplot(x='person_id', y='h_T_x', data=combined_df)
    # sns.violinplot(y='h_T_x', data=combined_df)
    # plt.title('head pose T_x')

    # plt.subplot(1, 3, 2)
    # # sns.violinplot(x='person_id', y='h_T_y', data=combined_df)
    # sns.violinplot(y='h_T_y', data=combined_df)
    # plt.title('head pose T_y')
    
    # plt.subplot(1, 3, 3)
    # # sns.violinplot(x='person_id', y='h_T_z', data=combined_df)
    # sns.violinplot(y='h_T_z', data=combined_df)
    # plt.title('head pose T_z')
    
    
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 3, 1)
    # # sns.violinplot(x='person_id', y='normalized_vector_x', data=combined_df)
    # sns.violinplot( y='normalized_vector_x', data=combined_df)
    # plt.title('gaze vector x')

    # plt.subplot(1, 3, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_y', data=combined_df)
    # sns.violinplot( y='normalized_vector_y', data=combined_df)
    # plt.title('gaze vector y')
    
    # plt.subplot(1, 3, 3)
    # # sns.violinplot(x='person_id', y='normalized_vector_z', data=combined_df)
    # sns.violinplot( y='normalized_vector_z', data=combined_df)
    # plt.title('gaze vector z')
    
    
    
    
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 2, 1)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    # sns.violinplot(y='normalized_vector_angle_pitch', data=combined_df)
    # plt.title('gaze angle_pitch')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍

    # plt.subplot(1, 2, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    # sns.violinplot(y='normalized_vector_angle_yaw', data=combined_df)
    # plt.title('gaze angle_yaw')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍
    
    # # 顯示圖表
    # plt.show()
    
    
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 2, 1)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    # sns.violinplot(y='normalized_h_angle_yaw', data=combined_df)
    # plt.title('head pose angle_pitch')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍

    # plt.subplot(1, 2, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    # sns.violinplot(y='normalized_h_angle_yaw', data=combined_df)
    # plt.title('head pose angle_yaw')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍
    
    # # 顯示圖表
    # plt.show()
    
    
    

    
    
    
    # data_x = combined_df['normalized_vector_angle_yaw'].values
    # data_y = combined_df['normalized_vector_angle_pitch'].values
    # data = np.vstack((data_x, data_y)).T

    # fig, ax = plt.subplots(figsize=(6, 6))

    # # Heatmap plot function with detailed grid
    # heatmap, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=np.arange(-80, 82, 1))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # im = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')

    # # Set grid
    # ax.set_xticks(np.arange(-80, 82, 1))  # Grid lines at each unit for x-axis
    # ax.set_yticks(np.arange(-80, 82, 1))  # Grid lines at each unit for y-axis
    # ax.grid(True, color='white', linestyle='-', linewidth=0.5)


    # # # # Set minor ticks for the grid lines
    # # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # # ax.yaxis.set_minor_locator(MultipleLocator(1))



    # # Customize ticks and labels
    # ax.set_xlabel('Yaw [deg]')
    # ax.set_ylabel('Pitch [deg]')
    # ax.set_xlim([-80, 81])
    # ax.set_ylim([-80, 81])
    # ax.set_title('Detailed Grid Heatmap')



    # # Add colorbar
    # plt.colorbar(im, ax=ax)
    # plt.tight_layout()
    # plt.show()
    
    
    
    data1 = combined_df_1['normalized_vector_angle_yaw'].values
    data2 = combined_df_2['normalized_vector_angle_yaw'].values
    data3 = combined_df_3['normalized_vector_angle_yaw'].values

    bins = np.linspace(-50, 50, 40)  # 40個bins，範圍是-100到100
    width = np.diff(bins)[0] / 2.0  # 每個條形的寬度，設置為bin寬度的一半

    # # 計算每組數據每個bin的百分比
    # hist1, bins1 = np.histogram(data1, bins, density=True)
    # hist2, bins2 = np.histogram(data2, bins, density=True)
    # hist3, bins3 = np.histogram(data3, bins, density=True)

    # # 將密度轉換為百分比
    # hist1_percent = hist1 * np.diff(bins1) * 100
    # hist2_percent = hist2 * np.diff(bins2) * 100
    # hist3_percent = hist3 * np.diff(bins3) * 100
    
    
    # 计算每组数据每个bin的百分比
    hist1, bins1 = np.histogram(data1, bins, density=False)
    hist2, bins2 = np.histogram(data2, bins, density=False)
    hist3, bins3 = np.histogram(data3, bins, density=False)

    # 计算总数
    total1 = len(data1)

    # 将数量转换为百分比，以data1的总量为基准
    hist1_percent = hist1 / total1 * 100
    hist2_percent = hist2 / total1 * 100  # 注意是除以total1，而不是total2
    hist3_percent = hist3 / total1 * 100  # 注意是除以total1，而不是total3

    # 創建一個2x1矩陣佈局的子圖
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # plt.bar(bins1[:-1], hist1_percent, width=np.diff(bins1), alpha=0.5, label='Data 1', align='edge')

    # # 繪製第二組數據的直方圖
    # plt.bar(bins2[:-1], hist2_percent, width=np.diff(bins2), alpha=0.5, label='Data 2', align='edge')
    
    
    # 繪製第一組數據的直方圖，左移width / 2
    plt.bar(bins1[:-1], hist1_percent, width=width, alpha=0.5, label='All', align='edge', edgecolor='black')

    # 繪製第二組數據的直方圖，右移width / 2
    plt.bar(bins2[:-1] + width / 2, hist2_percent, width=width/2, alpha=0.5, label='Phone', align='edge', edgecolor='black')
        
    # 繪製第二組數據的直方圖，右移width / 2
    plt.bar(bins3[:-1] , hist3_percent, width=width/2, alpha=0.5, label='Tablet', align='edge', edgecolor='black')

    # 添加標籤
    plt.xlabel('Horizontal gaze direction')
    plt.ylabel('Percentage of images')
    plt.title('Horizontal gaze direction distribution')
    plt.legend()

    # 顯示圖表
    plt.show()

    
    
    
    
    data_x = combined_df_1['normalized_vector_angle_yaw'].values
    data_y = combined_df_1['normalized_vector_angle_pitch'].values
    x_range = (-80, 80)
    y_range = (-80, 80)
    
    
    # 創建 2D 直方圖
    plt.hist2d(data_x, data_y, bins=160, range=[x_range, y_range], cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Yaw (deg)')
    plt.ylabel('Pitch (deg)')
    plt.title('Gaze Direction')
    plt.show()
    
    
    data_x = combined_df_2['normalized_vector_angle_yaw'].values
    data_y = combined_df_2['normalized_vector_angle_pitch'].values
    x_range = (-80, 80)
    y_range = (-80, 80)
    
    
    # 創建 2D 直方圖
    plt.hist2d(data_x, data_y, bins=160, range=[x_range, y_range], cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Yaw (deg)')
    plt.ylabel('Pitch (deg)')
    plt.title('Gaze Direction')
    plt.show()
    
    
    data_x = combined_df_3['normalized_vector_angle_yaw'].values
    data_y = combined_df_3['normalized_vector_angle_pitch'].values
    x_range = (-80, 80)
    y_range = (-80, 80)
    
    
    # 創建 2D 直方圖
    plt.hist2d(data_x, data_y, bins=160, range=[x_range, y_range], cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Yaw (deg)')
    plt.ylabel('Pitch (deg)')
    plt.title('Gaze Direction')
    plt.show()
    
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    sns.violinplot(y='normalized_vector_angle_pitch', data=combined_df_1)
    plt.title('gaze angle_pitch')
    plt.ylim(-80, 80)  # 設置 y 軸的範圍

    # plt.subplot(1, 2, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    # sns.violinplot(y='normalized_vector_angle_yaw', data=combined_df)
    # plt.title('gaze angle_yaw')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍
    
    # # 顯示圖表
    # plt.show()
    
    
    plt.subplot(1, 2, 2)
    sns.violinplot(x='normalized_vector_angle_yaw', data=combined_df_1)  # 使用 x 参数
    plt.title('gaze angle_yaw')
    plt.xlim(-80, 80)  # 设定 x 轴的范围
    plt.xlabel('normalized_vector_angle_yaw')  # 设置 x 轴标签
    plt.ylabel('')  # 移除 y 轴标签

    # 显示图表
    plt.show()
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    sns.violinplot(y='normalized_vector_angle_pitch', data=combined_df_2)
    plt.title('gaze angle_pitch')
    plt.ylim(-80, 80)  # 設置 y 軸的範圍

    # plt.subplot(1, 2, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    # sns.violinplot(y='normalized_vector_angle_yaw', data=combined_df)
    # plt.title('gaze angle_yaw')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍
    
    # # 顯示圖表
    # plt.show()
    
    
    plt.subplot(1, 2, 2)
    sns.violinplot(x='normalized_vector_angle_yaw', data=combined_df_2)  # 使用 x 参数
    plt.title('gaze angle_yaw')
    plt.xlim(-80, 80)  # 设定 x 轴的范围
    plt.xlabel('normalized_vector_angle_yaw')  # 设置 x 轴标签
    plt.ylabel('')  # 移除 y 轴标签

    # 显示图表
    plt.show()
    
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    sns.violinplot(y='normalized_vector_angle_pitch', data=combined_df_3)
    plt.title('gaze angle_pitch')
    plt.ylim(-80, 80)  # 設置 y 軸的範圍

    # plt.subplot(1, 2, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    # sns.violinplot(y='normalized_vector_angle_yaw', data=combined_df)
    # plt.title('gaze angle_yaw')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍
    
    # # 顯示圖表
    # plt.show()
    
    
    plt.subplot(1, 2, 2)
    sns.violinplot(x='normalized_vector_angle_yaw', data=combined_df_3)  # 使用 x 参数
    plt.title('gaze angle_yaw')
    plt.xlim(-80, 80)  # 设定 x 轴的范围
    plt.xlabel('normalized_vector_angle_yaw')  # 设置 x 轴标签
    plt.ylabel('')  # 移除 y 轴标签

    # 显示图表
    plt.show()
    
    
    
    
    
    
    
    
    # data_x = combined_df['mean_x'].values
    # data_y = combined_df['mean_y'].values
    # data = np.vstack((data_x, data_y)).T
    

    # # 绘制热图
    # fig, ax = plt.subplots(figsize=(12, 8))  # 调整 figsize 以适应新的分辨率

    # # Heatmap plot function with detailed grid
    # heatmap, xedges, yedges = np.histogram2d(data_x, data_y, bins=[np.arange(0, 1921, 10), np.arange(0, 1081, 10)])
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # im = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')

    # # Set grid
    # ax.set_xticks(np.arange(0, 1920, 200))  # Grid lines at every 100 units for x-axis
    # ax.set_yticks(np.arange(0, 1080, 100))   # Grid lines at every 50 units for y-axis
    # ax.grid(True, color='white', linestyle='-', linewidth=0.5)

    # # Customize ticks and labels
    # ax.set_xlabel('x [pixels]')
    # ax.set_ylabel('y [pixels]')
    # ax.set_xlim([0, 1920])
    # ax.set_ylim([0, 1080])
    # ax.set_title('Detailed Grid Heatmap')

    # # Add colorbar
    # plt.colorbar(im, ax=ax)
    # plt.tight_layout()
    # plt.show()
    
    
    
    # data_x = combined_df['normalized_h_angle_yaw'].values
    # data_y = combined_df['normalized_h_angle_pitch'].values
    # data = np.vstack((data_x, data_y)).T

    # fig, ax = plt.subplots(figsize=(6, 6))

    # # Heatmap plot function with detailed grid
    # heatmap, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=np.arange(-80, 82, 4))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # im = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')

    # # Set grid
    # ax.set_xticks(np.arange(-80, 82, 4))  # Grid lines at each unit for x-axis
    # ax.set_yticks(np.arange(-80, 82, 4))  # Grid lines at each unit for y-axis
    # ax.grid(True, color='white', linestyle='-', linewidth=0.5)


    # # # # Set minor ticks for the grid lines
    # # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # # ax.yaxis.set_minor_locator(MultipleLocator(1))



    # # Customize ticks and labels
    # ax.set_xlabel('Yaw [deg]')
    # ax.set_ylabel('Pitch [deg]')
    # ax.set_xlim([-80, 81])
    # ax.set_ylim([-80, 81])
    # ax.set_title('Detailed Grid Heatmap')



    # # Add colorbar
    # plt.colorbar(im, ax=ax)
    # plt.tight_layout()
    # plt.show()
    
    
    
    data1 = combined_df_1['normalized_h_angle_yaw'].values
    data2 = combined_df_2['normalized_h_angle_yaw'].values
    data3 = combined_df_3['normalized_h_angle_yaw'].values

    bins = np.linspace(-100, 100, 40)  # 40個bins，範圍是-100到100
    width = np.diff(bins)[0] / 2.0  # 每個條形的寬度，設置為bin寬度的一半

    # # 計算每組數據每個bin的百分比
    # hist1, bins1 = np.histogram(data1, bins, density=True)
    # hist2, bins2 = np.histogram(data2, bins, density=True)
    # hist3, bins3 = np.histogram(data3, bins, density=True)

    # # 將密度轉換為百分比
    # hist1_percent = hist1 * np.diff(bins1) * 100
    # hist2_percent = hist2 * np.diff(bins2) * 100
    # hist3_percent = hist3 * np.diff(bins3) * 100
    
    
    # 计算每组数据每个bin的百分比
    hist1, bins1 = np.histogram(data1, bins, density=False)
    hist2, bins2 = np.histogram(data2, bins, density=False)
    hist3, bins3 = np.histogram(data3, bins, density=False)

    # 计算总数
    total1 = len(data1)

    # 将数量转换为百分比，以data1的总量为基准
    hist1_percent = hist1 / total1 * 100
    hist2_percent = hist2 / total1 * 100  # 注意是除以total1，而不是total2
    hist3_percent = hist3 / total1 * 100  # 注意是除以total1，而不是total3

    # 創建一個2x1矩陣佈局的子圖
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # plt.bar(bins1[:-1], hist1_percent, width=np.diff(bins1), alpha=0.5, label='Data 1', align='edge')

    # # 繪製第二組數據的直方圖
    # plt.bar(bins2[:-1], hist2_percent, width=np.diff(bins2), alpha=0.5, label='Data 2', align='edge')
    
    
    # 繪製第一組數據的直方圖，左移width / 2
    plt.bar(bins1[:-1], hist1_percent, width=width, alpha=0.5, label='All', align='edge', edgecolor='black')

    # 繪製第二組數據的直方圖，右移width / 2
    plt.bar(bins2[:-1] + width / 2, hist2_percent, width=width/2, alpha=0.5, label='Phone', align='edge', edgecolor='black')
        
    # 繪製第二組數據的直方圖，右移width / 2
    plt.bar(bins3[:-1] , hist3_percent, width=width/2, alpha=0.5, label='Tablet', align='edge', edgecolor='black')

    # 添加標籤
    plt.xlabel('Horizontal head pose')
    plt.ylabel('Percentage of images')
    plt.title('Horizontal head pose distribution')
    plt.legend()

    # 顯示圖表
    plt.show()
    
    
    
    
    data_x = combined_df_1['normalized_h_angle_yaw'].values
    data_y = combined_df_1['normalized_h_angle_pitch'].values
    x_range = (-80, 80)
    y_range = (-80, 80)
    
    
    # 創建 2D 直方圖
    plt.hist2d(data_x, data_y, bins=160, range=[x_range, y_range], cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Yaw (deg)')
    plt.ylabel('Pitch (deg)')
    plt.title('Head Pose')
    plt.show()
    
    
    data_x = combined_df_2['normalized_h_angle_yaw'].values
    data_y = combined_df_2['normalized_h_angle_pitch'].values
    x_range = (-80, 80)
    y_range = (-80, 80)
    
    
    # 創建 2D 直方圖
    plt.hist2d(data_x, data_y, bins=160, range=[x_range, y_range], cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Yaw (deg)')
    plt.ylabel('Pitch (deg)')
    plt.title('Head Pose')
    plt.show()
    
    
    data_x = combined_df_3['normalized_h_angle_yaw'].values
    data_y = combined_df_3['normalized_h_angle_pitch'].values
    x_range = (-80, 80)
    y_range = (-80, 80)
    
    
    # 創建 2D 直方圖
    plt.hist2d(data_x, data_y, bins=160, range=[x_range, y_range], cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.xlabel('Yaw (deg)')
    plt.ylabel('Pitch (deg)')
    plt.title('Head Pose')
    plt.show()
    
    
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    sns.violinplot(y='normalized_h_angle_yaw', data=combined_df_1)
    plt.title('head pose angle_pitch')
    plt.ylim(-80, 80)  # 設置 y 軸的範圍

    # plt.subplot(1, 2, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    # sns.violinplot(y='normalized_h_angle_yaw', data=combined_df)
    # plt.title('head pose angle_yaw')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍
    
    # # 顯示圖表
    # plt.show()
    
    
    plt.subplot(1, 2, 2)
    sns.violinplot(x='normalized_h_angle_yaw', data=combined_df_1)  # 使用 x 参数
    plt.title('head pose angle_yaw')
    plt.xlim(-80, 80)  # 设定 x 轴的范围
    plt.xlabel('normalized_h_angle_yaw')  # 设置 x 轴标签
    plt.ylabel('')  # 移除 y 轴标签

    # 显示图表
    plt.show()
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    sns.violinplot(y='normalized_h_angle_yaw', data=combined_df_2)
    plt.title('head pose angle_pitch')
    plt.ylim(-80, 80)  # 設置 y 軸的範圍

    # plt.subplot(1, 2, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    # sns.violinplot(y='normalized_h_angle_yaw', data=combined_df)
    # plt.title('head pose angle_yaw')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍
    
    # # 顯示圖表
    # plt.show()
    
    
    plt.subplot(1, 2, 2)
    sns.violinplot(x='normalized_h_angle_yaw', data=combined_df_2)  # 使用 x 参数
    plt.title('head pose angle_yaw')
    plt.xlim(-80, 80)  # 设定 x 轴的范围
    plt.xlabel('normalized_h_angle_yaw')  # 设置 x 轴标签
    plt.ylabel('')  # 移除 y 轴标签

    # 显示图表
    plt.show()
    
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    # sns.violinplot(x='person_id', y='normalized_vector_angle_yaw', data=combined_df)
    sns.violinplot(y='normalized_h_angle_yaw', data=combined_df_3)
    plt.title('head pose angle_pitch')
    plt.ylim(-80, 80)  # 設置 y 軸的範圍

    # plt.subplot(1, 2, 2)
    # # sns.violinplot(x='person_id', y='normalized_vector_angle_pitch', data=combined_df)
    # sns.violinplot(y='normalized_h_angle_yaw', data=combined_df)
    # plt.title('head pose angle_yaw')
    # plt.ylim(-100, 100)  # 設置 y 軸的範圍
    
    # # 顯示圖表
    # plt.show()
    
    
    plt.subplot(1, 2, 2)
    sns.violinplot(x='normalized_h_angle_yaw', data=combined_df_3)  # 使用 x 参数
    plt.title('head pose angle_yaw')
    plt.xlim(-80, 80)  # 设定 x 轴的范围
    plt.xlabel('normalized_h_angle_yaw')  # 设置 x 轴标签
    plt.ylabel('')  # 移除 y 轴标签

    # 显示图表
    plt.show()
    
    



if __name__ == '__main__':
    main()
    