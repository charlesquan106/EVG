import h5py
import cv2
import numpy as np
import torch

    
with h5py.File('webcam_c.h5', 'r') as file:
    # 現在您可以訪問 H5 檔案中的資料集
    # dataset = file['face_PoG_tobii/validity']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_PoG_tobii/validity']  # 替換成您要訪問的資料集名稱
    # dataset = file['facial_landmarks/data']  # 替換成您要訪問的資料集名稱
    # dataset = file['head_rvec/data']  # 替換成您要訪問的資料集名稱
    # dataset = file['camera_transformation']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_PoG_tobii/data']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_o/data']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_o/validity']  # 替換成您要訪問的資料集名稱
    # dataset = file['inv_camera_transformation']  # 替換成您要訪問的資料集名稱
    # dataset = file['pixels_per_millimeter']  # 替換成您要訪問的資料集名稱
    # dataset = file['millimeters_per_pixel']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_g_tobii/data']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_R/data']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_R/validity']  # 替換成您要訪問的資料集名稱
    dataset = file['facial_landmarks/data']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_g_tobii/data']  # 替換成您要訪問的資料集名稱
    # dataset = file['face_h/data']  # 替換成您要訪問的資料集名稱

    
    
    
    # 讀取資料集的內容
    # data = dataset[:4]
    data = dataset[:]
    
    
    
    
    # data = np.rad2deg(np.array(data))
    # print(data.shape)
    # print(data)
    
    
    print(data[0][36])
    print(data[0][39])
    
    
    # origin = torch.tensor(data, dtype=torch.float32)
    
    # print(origin)
    # # tensor.tolist()

    
    # print(data)
    
