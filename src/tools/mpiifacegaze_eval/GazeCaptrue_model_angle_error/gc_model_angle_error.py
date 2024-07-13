"""
Copyright 2019 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys

import cv2
import h5py
import numpy as np

import argparse
import math

import collections
import time
from argparse import ArgumentParser

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mpiifacegaze_eval_lib.webcam import WebcamSource
from mpiifacegaze_eval_lib.utils_camera import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen



import matplotlib.pyplot as plt
import glob

import torch
import tqdm
from mpiifacegaze_eval_lib.utils.utils import AverageMeter 
from mpiifacegaze_eval_lib.utils.common import vector_to_pitchyaw
from mpiifacegaze_eval_lib.opts import opts
from mpiifacegaze_eval_lib.datasets.dataset_factory import get_dataset
from mpiifacegaze_eval_lib.models.model import create_model, load_model, save_model
from mpiifacegaze_eval_lib.models.decode import ctdet_gaze_decode
from mpiifacegaze_eval_lib.utils.post_process import ctdet_gaze_post_process
from mpiifacegaze_eval_lib.models.utils import _sigmoid
from mpiifacegaze_eval_lib.utils.image import get_affine_transform, affine_transform, transform_preds




import pathlib
import pandas as pd
import json




face_model_3d_coordinates = None

GazeCapture_info = []


output_txt_path = f'GazeCapture_anno_gaze_test_phone.txt'
output_person_specific_txt_path = f'GazeCapture_person_specific_test_phone.txt'


normalized_camera = {
    'focal_length': 1300,
    'distance': 600,
    'size': (256, 64),
}

norm_camera_matrix = np.array(
    [
        [normalized_camera['focal_length'], 0, 0.5*normalized_camera['size'][0]],  # noqa
        [0, normalized_camera['focal_length'], 0.5*normalized_camera['size'][1]],  # noqa
        [0, 0, 1],
    ],
    dtype=np.float64,
)


class Undistorter:

    _map = None
    _previous_parameters = None

    def __call__(self, image, camera_matrix, distortion, is_gazecapture=False):
        h, w, _ = image.shape
        all_parameters = np.concatenate([camera_matrix.flatten(),
                                         distortion.flatten(),
                                         [h, w]])
        if (self._previous_parameters is None
                or len(self._previous_parameters) != len(all_parameters)
                or not np.allclose(all_parameters, self._previous_parameters)):
            print('Distortion map parameters updated.')
            self._map = cv2.initUndistortRectifyMap(
                camera_matrix, distortion, R=None,
                newCameraMatrix=camera_matrix if is_gazecapture else None,
                size=(w, h), m1type=cv2.CV_32FC1)
            print('fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f' % (
                    camera_matrix[0, 0], camera_matrix[1, 1],
                    camera_matrix[0, 2], camera_matrix[1, 2]))
            self._previous_parameters = np.copy(all_parameters)

        # Apply
        return cv2.remap(image, self._map[0], self._map[1], cv2.INTER_LINEAR)


undistort = Undistorter()


def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2,
              color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx,
                                   eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out


def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to pitch (theta) and yaw (phi) angles."""
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


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


def data_normalization(dataset_name, dataset_path, group, output_path, person_id, model, person_device_info):
    

    # Prepare methods to organize per-entry outputs
    to_write = {}
    def add(key, value):  # noqa
        if key not in to_write:
            to_write[key] = [value]
        else:
            to_write[key].append(value)

    # Iterate through group (person_id)
    num_entries = next(iter(group.values())).shape[0]
    print("num_entries = ", num_entries)
    
    
    person_dir = '%s/%s' % (dataset_path,person_id)
    
    print("person_dir = ", person_dir)
    gt_info = json.load(open(os.path.join(person_dir, "dotInfo.json")))
    screen_info = json.load(open(os.path.join(person_dir, "screen.json")))
    
    for i in range(num_entries):
        # Perform data normalization   
        processed_entry = data_normalization_entry(dataset_name, dataset_path,
                                            group, i, model, gt_info, screen_info)

        # # Gather all of the person's data
        # add('pixels', processed_entry['patch'])
        # add('labels', np.concatenate([
        #     processed_entry['normalized_gaze_direction'],
        #     processed_entry['normalized_head_pose'],
        # ]))


        flattened_face_g = ' '.join(map(str, processed_entry['normalized_gaze_direction']))
        # flattened_face_h = ' '.join([str(item[0]) for item in face_h])
        flattened_head_h = ' '.join(map(str, processed_entry['head_pose']))
        flattened_norm_head_h = ' '.join(map(str, processed_entry['normalized_head_pose']))
        flattened_g_error = ' '.join(map(str, processed_entry['gaze_error']))
        
        
        
        GazeCapture_info.append(f"{flattened_head_h} {flattened_norm_head_h} {flattened_face_g} {flattened_g_error}")

    # if len(to_write) == 0:
    #     return

    # # Cast to numpy arrays
    # for key, values in to_write.items():
    #     to_write[key] = np.asarray(values)
    #     print('%s: ' % key, to_write[key].shape)

    # # Write to HDF
    # with h5py.File(output_path,
    #                'a' if os.path.isfile(output_path) else 'w') as f:
    #     if person_id in f:
    #         del f[person_id]
    #     group = f.create_group(person_id)
    #     for key, values in to_write.items():
    #         group.create_dataset(
    #             key, data=values,
    #             chunks=(
    #                 tuple([1] + list(values.shape[1:]))
    #                 if isinstance(values, np.ndarray)
    #                 else None
    #             ),
    #             compression='lzf',
    #         )


def data_normalization_entry(dataset_name, dataset_path, group, i , model, gt_info, screen_info):

    # Form original camera matrix
    fx, fy, cx, cy = group['camera_parameters'][i, :]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=np.float64)

    # Grab image
    image_path = '%s/%s' % (dataset_path,
                            group['file_name'][i].decode('utf-8'))
    # print(dataset_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = undistort(image, camera_matrix,
                      group['distortion_parameters'][i, :],
                      is_gazecapture=(dataset_name == 'GazeCapture'))
    
    image = image[:, :, ::-1]  # BGR to RGB
    
    
    
    ########### model preprocessing ##########
    
    
    mean = np.array([0.5893, 0.5006, 0.4467],
                   dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.2713, 0.2683, 0.2581],
                   dtype=np.float32).reshape(1, 1, 3)
    
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    img_height, img_width = img.shape[0], img.shape[1]
    
    input_h, input_w = 512, 512
    # input_h, input_w = opt.input_h, opt.input_w

    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    # if opt.keep_res:
    #     input_h = (img_height | opt.pad) + 1
    #     input_w = (img_width | opt.pad) + 1
    #     s = np.array([input_w, input_h], dtype=np.float32)
    # else:
    #     s = max(img_width, img_height) * 1.0
    #     input_h, input_w = input_h, input_w
      
    s = max(img_width, img_height) * 1.0
    # input_h, input_w = input_h, input_w
        
    # print("input_h , input_w = ", input_h,input_w )
        
        
    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                    (input_w, input_h),
                    flags=cv2.INTER_LINEAR)


    inp = (inp.astype(np.float32) / 255.)
    # if self.split == 'train' and not self.opt.no_color_aug:
    #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

    inp = (inp - mean) / std

    inp_output = inp

    inp = inp.transpose(2, 0, 1)

    inp = torch.from_numpy(inp).unsqueeze(0)

    inp = inp.to(device)


    ################ vp  ################
    # vp_width, vp_height = opt.vp_w, opt.vp_h
    vp_width, vp_height = 2400, 2400
    # vp = np.zeros((vp_height, vp_width , 1), dtype=np.float32)
    # vp = vp.transpose(1,2,0)

    ############ check  sc_point ############

    sc_gazepoint = [gt_info["XPts"][i], gt_info["YPts"][i]]
    sc_width = screen_info["W"][i]
    sc_height = screen_info["H"][i]
    camera_screen_x_offset = 0 
    camera_screen_y_offset = 0
    # vp_gazepoint = np.array([(vp_width/2)+(sc_gazepoint[0]-(sc_width/2))+ camera_screen_x_offset, (vp_height/2)+(sc_gazepoint[1]-(sc_height/2))+camera_screen_y_offset], dtype=np.float32)

    # print("vp_gazepoint =",vp_gazepoint)

    vp_c = np.array([ vp_width / 2.0, vp_height/ 2.0], dtype=np.float32)
    # if opt.keep_res:
    #     # trans_output_h = (vp_height | self.opt.pad) + 1
    #     # trans_output_w = (vp_width | self.opt.pad) + 1
    #     # vp_s = np.array([trans_output_w, trans_output_h], dtype=np.float32)
    #     trans_output_h = (vp_height | opt.pad) + 1
    #     trans_output_w = (vp_width | opt.pad) + 1
    #     vp_s = np.array([vp_width, vp_height], dtype=np.float32)
    # else:
    #     # vp_s = max(vp_width, vp_height) * 1.0
    #     vp_s = np.array([vp_width, vp_height], dtype=np.float32)

    vp_s = np.array([vp_width, vp_height], dtype=np.float32)


    # print("inp = ", inp.shape)
    # cv2.imshow("Tracking", inp)

    # cv2.waitKey(5000)
    
    
    
    # key = cv2.waitKey(0) & 0xF
    
    
    # if key == ord('q'):
    #     break
    
    # cv2.imshow("Tracking", img)
    # cv2.destroyAllWindows()



    ################ model  ################

    outputs = model(inp)
    
    output = outputs[-1]
    hm = output['hm'].sigmoid_()
    reg = output['reg']
    dets = ctdet_gaze_decode(hm, reg=reg, K=1)
    
    
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    
    # print("dets:", dets[0][0][0], "  ",dets[0][0][1] )

    # print("output['hm'].shape[2] /  output['hm'].shape[3] : ", output['hm'].shape[2], "   ", output['hm'].shape[3]  )
    # print("vp_c /  vp_s : ", vp_c, "   ", vp_s  )
    
    dets_out = transform_preds(dets[0, :, 0:2], 
                                vp_c,vp_s,
                                (output['hm'].shape[3], output['hm'].shape[2]))
    
    # print(f"dets_out: {dets_out}")
    dets_org_coord = dets_out
    # print(f"dets_org_coord: {dets_org_coord}")
    
    dets_modified = (int((dets_org_coord[0][0] + (sc_width/2) - (vp_width/2))),int((dets_org_coord[0][1]+ (sc_height/2)-(vp_height/2))))
    
    # print("dets_modified: " , dets_modified)
    
    
    ########## XPts / YPts -> XCam / YCam #########
    
    Device_camera_to_screen_x_mm = float(person_device_info['DeviceCameraToScreenXMm'])
    Device_screen_width_mm = float(person_device_info['DeviceScreenWidthMm'])
    Device_camera_to_screen_y_mm = float(person_device_info['DeviceCameraToScreenYMm'])
    Device_screen_height_mm = float(person_device_info['DeviceScreenHeightMm'])
    
    
    Orientation = screen_info["Orientation"][i]
    
    if Orientation == 1:
        # XCam = person_device_info['DeviceCameraToScreenXMm'] - (dets_modified * (person_device_info['DeviceScreenWidthMm']/sc_width))
        # YCam = person_device_info['DeviceCameraToScreenYMm'] - (dets_modified * (person_device_info['DeviceScreenWidthMm']/sc_width))  
        g_t_predict = np.array([[-(-Device_camera_to_screen_x_mm + (dets_modified[0] * (Device_screen_width_mm/sc_width)))], 
                                [-(-Device_camera_to_screen_y_mm - (dets_modified[1] * (Device_screen_height_mm/sc_height)))], 
                                [-0.0]])
          
    elif Orientation == 2:
        
        g_t_predict = np.array([[-(Device_camera_to_screen_x_mm - Device_screen_width_mm + (dets_modified[0] * (Device_screen_width_mm/sc_width)))], 
                                [-(Device_camera_to_screen_y_mm + Device_screen_height_mm - (dets_modified[1] * (Device_screen_height_mm/sc_height)))], 
                                [-0.0]])

    elif Orientation == 3:
        # with home button on the right
        # width & height switch
        g_t_predict = np.array([[-(Device_camera_to_screen_y_mm + (dets_modified[0] * (Device_screen_height_mm/sc_height)))], 
                                [-(-Device_camera_to_screen_x_mm + Device_screen_width_mm - (dets_modified[1] * (Device_screen_width_mm/sc_width)))], 
                                [-0.0]])
    elif Orientation == 4:
        # with home button on the left
        # width & height switch
        g_t_predict = np.array([[-(-Device_camera_to_screen_y_mm - Device_screen_height_mm + (dets_modified[0] * (Device_screen_height_mm/sc_height)))], 
                                [-(Device_camera_to_screen_x_mm - (dets_modified[1] * (Device_screen_width_mm/sc_width)))], 
                                [-0.0]])
    
    
    #########################
    # print(image_path)

    # Calculate rotation matrix and euler angles
    rvec = group['head_pose'][i, :3].reshape(3, 1)
    tvec = group['head_pose'][i, 3:].reshape(3, 1)
    rotate_mat, _ = cv2.Rodrigues(rvec)

    # Take mean face model landmarks and get transformed 3D positions
    landmarks_3d = np.matmul(rotate_mat, face_model_3d_coordinates.T).T
    landmarks_3d += tvec.T

    # Gaze-origin (g_o) and target (g_t)
    g_o = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
    g_o = g_o.reshape(3, 1)
    g_t = group['3d_gaze_target'][i, :].reshape(3, 1)
    g = g_t - g_o
    g /= np.linalg.norm(g)
    g_pitchyaw = g.reshape(1, -1)
    g_pitchyaw = abs(vector_to_pitchyaw(g_pitchyaw))
    
    # print("g  = ", g)
    # print("g_pitchyaw  = ", g_pitchyaw)
    # print("g_pitchyaw (deg)  = ", np.degrees(g_pitchyaw[0]))
    
    # Gaze-origin (g_o) and target (g_t_predict)
    g_predict = g_t_predict - g_o
    g_predict /= np.linalg.norm(g_predict)
    g_predict_pitchyaw = g_predict.reshape(1, -1)
    g_predict_pitchyaw = abs(vector_to_pitchyaw(g_predict_pitchyaw))
    

    # print("g_predict  = ", g_predict)
    # print("g_predict_pitchyaw  = ", g_predict_pitchyaw)
    # print("g_predict_pitchyaw (deg)  = ", np.degrees(g_predict_pitchyaw[0]))
    
    g_error = abs(np.degrees(g_pitchyaw[0])- np.degrees(g_predict_pitchyaw[0]))
    # print("direction_error = ", direction_error)
    # print("g_error (pitch) = ", g_error[0])
    # print("g_error (yaw) = ", g_error[1])
    
    if(g_error[1]) > 20:
        
        print("-------------------------------------")
        print("Orientation = ", Orientation)
        print("g  = ", g)
        print("g_pitchyaw  = ", g_pitchyaw)
        print("g_pitchyaw (deg)  = ", np.degrees(g_pitchyaw[0]))
        
        print("g_predict  = ", g_predict)
        print("g_predict_pitchyaw  = ", g_predict_pitchyaw)
        print("g_predict_pitchyaw (deg)  = ", np.degrees(g_predict_pitchyaw[0]))

    
    
    
    # Code below is an adaptation of code by Xucong Zhang
    # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

    # actual distance between gaze origin and original camera
    distance = np.linalg.norm(g_o)
    z_scale = normalized_camera['distance'] / distance
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = z_scale

    hRx = rotate_mat[:, 0]
    forward = (g_o / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    # transformation matrix
    W = np.dot(np.dot(norm_camera_matrix, S),
               np.dot(R, np.linalg.inv(camera_matrix)))

    ow, oh = normalized_camera['size']
    patch = cv2.warpPerspective(image, W, (ow, oh))  # image normalization

    R = np.asmatrix(R)

    # Correct head pose
    h = np.array([np.arcsin(rotate_mat[1, 2]),
                  np.arctan2(rotate_mat[0, 2], rotate_mat[2, 2])])
    head_mat = R * rotate_mat
    n_h = np.array([np.arcsin(head_mat[1, 2]),
                    np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    # Correct gaze
    n_g = R * g
    n_g /= np.linalg.norm(n_g)
    n_g = vector_to_pitchyaw(-n_g.T).flatten()

    # Basic visualization for debugging purposes
    # if i % 50 == 0:
    #     to_visualize = cv2.equalizeHist(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))
    #     to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.25 * oh), n_g,
    #                              length=80.0, thickness=1)
    #     to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.75 * oh), n_h,
    #                              length=40.0, thickness=3, color=(0, 0, 0))
    #     to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.75 * oh), n_h,
    #                              length=40.0, thickness=1,
    #                              color=(255, 255, 255))
    #     cv2.imshow('normalized_patch', to_visualize)
    #     cv2.waitKey(1)

    return {
        'patch': patch.astype(np.uint8),
        'gaze_direction': g.astype(np.float32),
        'gaze_origin': g_o.astype(np.float32),
        'gaze_target': g_t.astype(np.float32),
        'gaze_error': g_error.astype(np.float32),
        'head_pose': h.astype(np.float32),
        'normalization_matrix': np.transpose(R).astype(np.float32),
        'normalized_gaze_direction': n_g.astype(np.float32),
        'normalized_head_pose': n_h.astype(np.float32),
    }

    

if __name__ == '__main__':
    
    output_txt_path = f'GazeCapture_anno_gaze_test_phone.txt'
    output_person_specific_txt_path = f'GazeCapture_person_specific_test_phone.txt'
    model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/gazecapture/resdcnface_18/gaze_gazecapture_all_no_scp_f001_2/model_37.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--person_specific_txt_path', type=str, required=True)
    parser.add_argument('-m','--model_path', type=str, required=True)
    parser.add_argument('-t','--txt_path', type=str, required=True)
    args = parser.parse_args()

    output_txt_path = os.path.join(os.getcwd(),"output", args.txt_path)
    print("output_txt_path = ", output_txt_path)
    output_person_specific_txt_path = os.path.join(os.getcwd(),"person_specific", args.person_specific_txt_path)
    model_path = args.model_path
    
    
    # Grab SFM coordinates and store
    face_model_fpath = './sfm_face_coordinates.npy'
    face_model_3d_coordinates = np.load(face_model_fpath)

    # Preprocess some datasets
    output_dir = '/home/owenserver/storage/Datasets/GazeCapture/ProcessData/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    datasets = {
        # 'MPIIGaze': {
        #     # Path to the MPIIFaceGaze dataset
        #     # Sub-folders names should consist of person IDs, for example:
        #     # p00, p01, p02, ...
        #     'input-path': '/media/wookie/WookExt4/datasets/MPIIFaceGaze',

        #     # A supplementary HDF file with preprocessing data,
        #     # as provided by us. See grab_prerequisites.bash
        #     'supplementary': './MPIIFaceGaze_supplementary.h5',

        #     # Desired output path for the produced HDF
        #     'output-path': output_dir + '/MPIIGaze.h5',
        # },
        'GazeCapture': {
            # Path to the GazeCapture dataset
            # Sub-folders names should consist of person IDs, for example:
            # 00002, 00028, 00141, ...
            'input-path': '/home/owenserver/storage/Datasets/GazeCapture/Data',

            # A supplementary HDF file with preprocessing data,
            # as provided by us. See grab_prerequisites.bash
            # 'supplementary': './GazeCapture_supplementary.h5',
            'supplementary': '/home/owenserver/Python/faze_preprocess/GazeCapture_supplementary.h5',

            # Desired output path for the produced HDF
            'output-path': output_dir + '/GazeCapture.h5',
        },
    }
    
    
    with open(output_person_specific_txt_path, 'r', encoding='utf-8') as file:
        person_specific_device_list = [line.strip() for line in file if line.strip()]
        
    # print(person_specific_device_list)

    
    ###############  model  #################
    heads = {'hm': 1,'reg':2}
    head_conv = 64
    print('Creating model...')
    arch = 'resdcn_18'
    model = create_model(arch, heads, head_conv)
    
    
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/gazecapture/resdcn_18/gaze_gazecapture_all_no_scp_pl01/model_58.pth"
    # model_path = "/home/owenserver/Python/CenterNet_gaze/exp/ctdet_gazeface/gazecapture/resdcnface_18/gaze_gazecapture_all_no_scp_f001_2/model_37.pth"
    model = load_model(model, model_path)
    
    model.eval()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda")
    # device = torch.device(opt.device)
    
    model = model.to(device)
    
    ##########################################
    
    for dataset_name, dataset_spec in datasets.items():
        # Perform the data normalization
        with h5py.File(dataset_spec['supplementary'], 'r') as f:
            for person_id, group in f.items():
                # print('')
                # print('Processing %s/%s' % (dataset_name, person_id))
                

                ########## loadAppleDeviceData ###########
                
                device_csv_dir = "/home/owenserver/storage/Datasets/GazeCapture/apple_device_data.csv"
                device_info = loadAppleDeviceData(device_csv_dir)
                
                # persons = os.listdir(dataset_path)
                # persons.sort()
                
                
                specify_device = "tablet"
                
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
                        
                        
                        
                        
                person_dir = os.path.join(dataset_spec['input-path'], person_id)
                person_info = json.load(open(os.path.join(person_dir, "info.json")))
                

                splited_set = person_info["Dataset"]
                devices = person_info["DeviceName"]
                
                # print("person_id = ", person_id , " / Dataset = ", splited_set  ," / DeviceName = ", devices )
                for index in range(len(device_info)) : 
                    if index == 0:
                        continue
                    if device_info[index]['DeviceName'] == devices: 
                        person_device_info = device_info[index]
                        break
                    

                    
                    # print(datatype)
                    
                    # if splited_set == datatype  or datatype == None :
                    #     if devices in specify_device_list or devices == None :
                    #         print(devices)
                    #         ImageProcessing_Person(person_dir, output_img_dir, output_xml_dir, person, person_device_info)
                
                
                ##########################################
                
                
                
                if person_id  not in person_specific_device_list : continue
                print('')
                print('Processing %s/%s' % (dataset_name, person_id))
                data_normalization(dataset_name,
                                   dataset_spec['input-path'],
                                   group,
                                   dataset_spec['output-path'],person_id, model , person_device_info)
                
    with open(output_txt_path, 'w', encoding='utf-8') as output_file:
        for info in GazeCapture_info:
            output_file.write(info + '\n')
