from models.utils import _sigmoid

import numpy as np

import torch
val = torch.tensor(-2.210591)

val_sigmoid = _sigmoid(val)

print(val_sigmoid)


# valid_ids = np.arange(1, 21, dtype=np.int32)
# print(valid_ids)

# cat_ids = {v: i for i, v in enumerate(valid_ids)}
# print(cat_ids)


# valid_ids = [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
#     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
#     24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
#     37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
#     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
#     58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
#     72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
#     82, 84, 85, 86, 87, 88, 89, 90]
# cat_ids = {v: i for i, v in enumerate(valid_ids)}
# print(cat_ids)

# 1:0 -> valid_id = 1 是屬於類別0 (__background__)
# 90:79 -> valid_id = 90 是屬於類別79 (toothbrush)


import cv2
import numpy as np


def affine_transform(pt, t):
    pt_x = float(pt[0])
    pt_y = float(pt[1])
    new_pt = np.array([pt_x, pt_y, 1.], dtype=np.float32).T
    # new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

# # 原始圖像中的三個點的坐標
# src_pts = np.float32([[50, 50], [200, 50], [50, 200]])

# # 目標圖像中對應的三個點的坐標
# dst_pts = np.float32([[70, 80], [210, 60], [70, 220]])

# 原始圖像中的三個點的坐標
src_pts = np.float32([[1280, 800],
 [1280, -480],
 [   0, -480]])

# 目標圖像中對應的三個點的坐標
dst_pts = np.float32([[64, 64],
 [64 , 0],
 [ 0 , 0]])



affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)

print("affine_matrix:")
print(affine_matrix)

# 擴展仿射變換矩陣的大小為 3x3
extended_affine_matrix = np.vstack((affine_matrix, [0, 0, 1]))

# 計算擴展矩陣的逆矩陣
inverse_extended_affine_matrix = np.linalg.inv(extended_affine_matrix)

# 提取原始變換的逆矩陣部分
inverse_affine_matrix = inverse_extended_affine_matrix[:2, :]

vp_gazepoint = [1708 , 640]

vp_gazepoint_output = affine_transform(vp_gazepoint, affine_matrix)

print("vp_gazepoint_output:")
print(vp_gazepoint_output)

point_to_transform = np.float32([[85.4 , 56]])
# point_to_transform = np.float32(vp_gazepoint_output)

print("Inverse Affine Transformation Matrix:")
print(inverse_affine_matrix)

reversed_point = cv2.transform(point_to_transform.reshape(1, -1, 2), inverse_affine_matrix)
reversed_point = reversed_point.reshape(-1, 2)

print("Reversed Point:")
print(reversed_point)

