"""Copyright 2020 ETH Zurich, Seonwook Park

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# from config_default import DefaultConfig
from typing import Tuple

# config = DefaultConfig()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def vector_to_pitchyaw(a):
    if a.shape[1] == 2:
        return a
    elif a.shape[1] == 3:
        a = a.view(-1, 3)
        norm_a = torch.div(a, torch.norm(a, dim=1).view(-1, 1) + 1e-7)
        return torch.stack([
            torch.asin(norm_a[:, 1]),
            torch.atan2(norm_a[:, 0], norm_a[:, 2]),
        ], dim=1)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def pitchyaw_to_rotation(a):
    if a.shape[1] == 3:
        a = vector_to_pitchyaw(a)

    cos = torch.cos(a)
    sin = torch.sin(a)
    ones = torch.ones_like(cos[:, 0])
    zeros = torch.zeros_like(cos[:, 0])
    matrices_1 = torch.stack([ones, zeros, zeros,
                              zeros, cos[:, 0], sin[:, 0],
                              zeros, -sin[:, 0], cos[:, 0]
                              ], dim=1)
    matrices_2 = torch.stack([cos[:, 1], zeros, sin[:, 1],
                              zeros, ones, zeros,
                              -sin[:, 1], zeros, cos[:, 1]
                              ], dim=1)
    matrices_1 = matrices_1.view(-1, 3, 3)
    matrices_2 = matrices_2.view(-1, 3, 3)
    matrices = torch.matmul(matrices_2, matrices_1)
    return matrices


def rotation_to_vector(a):
    assert(a.ndim == 3)
    assert(a.shape[1] == a.shape[2] == 3)
    frontal_vector = torch.cat([
        torch.zeros_like(a[:, :2, 0]).reshape(-1, 2, 1),
        torch.ones_like(a[:, 2, 0]).reshape(-1, 1, 1),
    ], axis=1)
    return torch.matmul(a, frontal_vector)


def apply_transformation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
        # print("pitchyaw_to_vector")
    vec = vec.reshape(-1, 3, 1)
    # print("vec.shape : ",vec.shape)
    h_vec = F.pad(vec, pad=(0, 0, 0, 1), value=1.0)
    # print("h_vec.shape : ",h_vec.shape)
    # print("h_vec : ",h_vec)
    # print("T.shape : ",T.shape)
    # print("T : ",T)
    return torch.matmul(T, h_vec)[:, :3, 0]


def apply_rotation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
        
        print(vec)
    vec = vec.reshape(-1, 3, 1)
    print(vec.shape)
    R = T[:, :3, :3]
    return torch.matmul(R, vec).reshape(-1, 3)


nn_plane_normal = None
nn_plane_other = None


def get_intersect_with_zero(o, g):
    """Intersects a given gaze ray (origin o and direction g) with z = 0."""
    global nn_plane_normal, nn_plane_other
    if nn_plane_normal is None:
        nn_plane_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
        nn_plane_other = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).view(1, 3, 1)

    # Define plane to intersect with
    n = nn_plane_normal
    a = nn_plane_other
    g = g.view(-1, 3, 1)
    o = o.view(-1, 3, 1)
    numer = torch.sum(torch.mul(a - o, n), dim=1)

    # Intersect with plane using provided 3D origin
    denom = torch.sum(torch.mul(g, n), dim=1) + 1e-7
    t = torch.div(numer, denom).view(-1, 1, 1)
    return (o + torch.mul(t, g))[:, :2, 0]

def calculate_combined_gaze_direction_normalize(avg_origin, avg_PoG, camera_transformation,face_R):
    # NOTE: PoG is assumed to be in mm and in the screen-plane
    avg_PoG_3D = F.pad(avg_PoG, (0, 1))
    # print("avg_PoG.shape = ", avg_PoG.shape)

    # Bring to camera-specific coordinate system, where the origin is
    avg_PoG_3D = apply_transformation(camera_transformation, avg_PoG_3D)
    direction = avg_PoG_3D - avg_origin
      
    list_of_matrices = []
    
    for i in range(avg_origin.shape[0]):

        R = face_R[i]
        direction_normalized = direction[i] 
        
        direction_normalized = np.dot(R, direction_normalized)
        direction_normalized = direction_normalized/np.linalg.norm(direction_normalized)
        
        list_of_matrices.append(direction_normalized)
        
    direction = np.stack(list_of_matrices)

    direction = direction.reshape(-1, 3, 1)
    direction = torch.tensor(direction)
    
    # # Negate gaze vector back (to user perspective)
    direction = -direction

    list_of_matrices = []
    for i in range(direction.shape[0]):
        direction_single = direction[i].reshape(3)
        direction_single = direction[i].reshape(1,-1)
        direction_single = vector_to_pitchyaw(direction_single)
        
        
        list_of_matrices.append(direction_single)

    direction = torch.cat(list_of_matrices, dim=0)
    
    return direction



def calculate_combined_gaze_direction(avg_origin, avg_PoG, head_rotation,
                                      camera_transformation):
    # NOTE: PoG is assumed to be in mm and in the screen-plane
    avg_PoG_3D = F.pad(avg_PoG, (0, 1))

    # Bring to camera-specific coordinate system, where the origin is
    avg_PoG_3D = apply_transformation(camera_transformation, avg_PoG_3D)
    direction = avg_PoG_3D - avg_origin
    
    
    # print(head_rotation.shape)

    # Rotate gaze vector back
    direction = direction.reshape(-1, 3, 1)
    
    direction = torch.matmul(head_rotation, direction)

    # Negate gaze vector back (to user perspective)
    direction = -direction

    direction = vector_to_pitchyaw(direction)
    return direction


def calculate_combined_gaze_direction_no_h(avg_origin, avg_PoG,
                                      camera_transformation):
    # NOTE: PoG is assumed to be in mm and in the screen-plane
    avg_PoG_3D = F.pad(avg_PoG, (0, 1))

    # Bring to camera-specific coordinate system, where the origin is
    avg_PoG_3D = apply_transformation(camera_transformation, avg_PoG_3D)
    direction = avg_PoG_3D - avg_origin
    
    
    # print(head_rotation.shape)

    # Rotate gaze vector back
    direction = direction.reshape(-1, 3, 1)
    
    # print(direction.shape)
    # print(head_rotation.shape)
    # direction = torch.matmul(head_rotation, direction)
    
    # print(direction.shape)

    # # Negate gaze vector back (to user perspective)
    # direction = -direction

    direction = vector_to_pitchyaw(direction)
    return direction


# def to_screen_coordinates(origin, direction, rotation, reference_dict):
#     direction = pitchyaw_to_vector(direction)

#     print(direction.shape)
#     print(direction)
#     # Negate gaze vector back (to camera perspective)
#     direction = -direction

#     # De-rotate gaze vector
#     inv_rotation = torch.transpose(rotation, 1, 2)
#     direction = direction.reshape(-1, 3, 1)
#     direction = torch.matmul(inv_rotation, direction)

#     # Transform values
#     inv_camera_transformation = reference_dict['inv_camera_transformation']
#     direction = apply_rotation(inv_camera_transformation, direction)
#     origin = apply_transformation(inv_camera_transformation, origin)

#     # Intersect with z = 0
#     recovered_target_2D = get_intersect_with_zero(origin, direction)
#     PoG_mm = recovered_target_2D

#     # Convert back from mm to pixels
#     ppm_w = reference_dict['pixels_per_millimeter'][:, 0]
#     ppm_h = reference_dict['pixels_per_millimeter'][:, 1]
#     PoG_px = torch.stack([
#         torch.clamp(recovered_target_2D[:, 0] * ppm_w,
#                     0.0, float(config.actual_screen_size[0])),
#         torch.clamp(recovered_target_2D[:, 1] * ppm_h,
#                     0.0, float(config.actual_screen_size[1]))
#     ], axis=-1)

#     return PoG_mm, PoG_px

# def to_screen_coordinates_inv(origin, direction, rotation, inv_camera_transformation):
#     direction = pitchyaw_to_vector(direction)

#     # print(direction.shape)
#     print("direction : ", direction)
#     # Negate gaze vector back (to camera perspective)
#     direction = -direction

#     # De-rotate gaze vector
#     inv_rotation = torch.transpose(rotation, 1, 2)
#     direction = direction.reshape(-1, 3, 1)
#     direction = torch.matmul(inv_rotation, direction)

#     # Transform values
#     inv_camera_transformation = inv_camera_transformation
    
#     # print("inv_camera_transformation.shape :",inv_camera_transformation.shape)
#     direction = apply_rotation(inv_camera_transformation, direction)
#     origin = apply_transformation(inv_camera_transformation, origin)

#     # Intersect with z = 0
#     recovered_target_2D = get_intersect_with_zero(origin, direction)
#     PoG_mm = recovered_target_2D

#     # Convert back from mm to pixels
    
#     # points[:, 0] = points[:, 0]*  3.471971
#     # points[:, 1] = points[:, 1]*  3.472669
#     ppm_w = 3.471971
#     ppm_h = 3.472669
#     PoG_px = torch.stack([
#         torch.clamp(recovered_target_2D[:, 0] * ppm_w,
#                     0.0, float(config.actual_screen_size[0])),
#         torch.clamp(recovered_target_2D[:, 1] * ppm_h,
#                     0.0, float(config.actual_screen_size[1]))
#     ], axis=-1)

#     return PoG_mm, PoG_px


def apply_offset_augmentation(gaze_direction, head_rotation, kappa, inverse_kappa=False):
    gaze_direction = pitchyaw_to_vector(gaze_direction)

    # Negate gaze vector back (to camera perspective)
    gaze_direction = -gaze_direction

    # De-rotate gaze vector
    inv_head_rotation = torch.transpose(head_rotation, 1, 2)
    gaze_direction = gaze_direction.reshape(-1, 3, 1)
    gaze_direction = torch.matmul(inv_head_rotation, gaze_direction)

    # Negate gaze vector back (to user perspective)
    gaze_direction = -gaze_direction

    # Apply kappa to frontal vector [0 0 1]
    kappa_vector = pitchyaw_to_vector(kappa).reshape(-1, 3, 1)
    if inverse_kappa:
        kappa_vector = torch.cat([
            -kappa_vector[:, :2, :], kappa_vector[:, 2, :].reshape(-1, 1, 1),
        ], axis=1)

    # Apply head-relative gaze to rotated frontal vector
    head_relative_gaze_rotation = pitchyaw_to_rotation(vector_to_pitchyaw(gaze_direction))
    gaze_direction = torch.matmul(head_relative_gaze_rotation, kappa_vector)

    # Negate gaze vector back (to camera perspective)
    gaze_direction = -gaze_direction

    # Rotate gaze vector back
    gaze_direction = gaze_direction.reshape(-1, 3, 1)
    gaze_direction = torch.matmul(head_rotation, gaze_direction)

    # Negate gaze vector back (to user perspective)
    gaze_direction = -gaze_direction

    gaze_direction = vector_to_pitchyaw(gaze_direction)
    return gaze_direction


heatmap_xs = None
heatmap_ys = None
heatmap_alpha = None


# def make_heatmap(centre, sigma):
#     global heatmap_xs, heatmap_ys, heatmap_alpha
#     w, h = config.gaze_heatmap_size
#     if heatmap_xs is None:
#         xs = np.arange(0, w, step=1, dtype=np.float32)
#         ys = np.expand_dims(np.arange(0, h, step=1, dtype=np.float32), -1)
#         heatmap_xs = torch.tensor(xs).to(device)
#         heatmap_ys = torch.tensor(ys).to(device)
#     heatmap_alpha = -0.5 / (sigma ** 2)
#     cx = (w / config.actual_screen_size[0]) * centre[0]
#     cy = (h / config.actual_screen_size[1]) * centre[1]
#     heatmap = torch.exp(heatmap_alpha * ((heatmap_xs - cx)**2 + (heatmap_ys - cy)**2))
#     heatmap = 1e-8 + heatmap  # Make the zeros non-zero (remove collapsing issue)
#     return heatmap.unsqueeze(0)  # make it (1 x H x W) in shape


# def batch_make_heatmaps(centres, sigma):
#     return torch.stack([make_heatmap(centre, sigma) for centre in centres], axis=0)


# gaze_history_map_decay_per_ms = None


# def make_gaze_history_map(history_timestamps, heatmaps, validities):
#     # NOTE: heatmaps has dimensions T x H x W
#     global gaze_history_map_decay_per_ms
#     target_timestamp = history_timestamps[torch.nonzero(history_timestamps)][-1]
#     output_heatmap = torch.zeros_like(heatmaps[0])
#     if gaze_history_map_decay_per_ms is None:
#         gaze_history_map_decay_per_ms = \
#             torch.tensor(config.gaze_history_map_decay_per_ms).to(device)

#     for timestamp, heatmap, validity in zip(history_timestamps, heatmaps, validities):

#         if timestamp == 0:
#             continue

#         # Calculate difference in time in milliseconds
#         diff_timestamp = (target_timestamp - timestamp) * 1e-6
#         assert(diff_timestamp >= 0)

#         # Weights for later weighted average
#         time_based_weight = torch.pow(gaze_history_map_decay_per_ms, diff_timestamp).view(1, 1)

#         # Keep if within time window
#         output_heatmap = output_heatmap + validity.float() * time_based_weight.detach() * heatmap

#     return output_heatmap


# def batch_make_gaze_history_maps(history_timestamps, heatmaps, validity):
#     # NOTE: timestamps is a tensor, heatmaps is a list of tensors
#     batch_size = history_timestamps.shape[0]
#     history_len = len(heatmaps)
#     return torch.stack([
#         make_gaze_history_map(
#             history_timestamps[b, :history_len],
#             [h[b, :] for h in heatmaps],
#             validity[b, :history_len],
#         )
#         for b in range(batch_size)
#     ], axis=0)


softargmax_xs = None
softargmax_ys = None


# def soft_argmax(heatmaps):
#     global softargmax_xs, softargmax_ys
#     if softargmax_xs is None:
#         # Assume normalized coordinate [0, 1] for numeric stability
#         w, h = config.gaze_heatmap_size
#         ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
#                                      np.linspace(0, 1.0, num=h, endpoint=True),
#                                      indexing='xy')
#         ref_xs = np.reshape(ref_xs, [1, h*w])
#         ref_ys = np.reshape(ref_ys, [1, h*w])
#         softargmax_xs = torch.tensor(ref_xs.astype(np.float32)).to(device)
#         softargmax_ys = torch.tensor(ref_ys.astype(np.float32)).to(device)
#     ref_xs, ref_ys = softargmax_xs, softargmax_ys

#     # Yield softmax+integrated coordinates in [0, 1]
#     n, _, h, w = heatmaps.shape
#     assert(w == config.gaze_heatmap_size[0])
#     assert(h == config.gaze_heatmap_size[1])
#     beta = 1e2
#     x = heatmaps.view(-1, h*w)
#     x = F.softmax(beta * x, dim=-1)
#     lmrk_xs = torch.sum(ref_xs * x, axis=-1)
#     lmrk_ys = torch.sum(ref_ys * x, axis=-1)

#     # Return to actual coordinates ranges
#     pixel_xs = torch.clamp(config.actual_screen_size[0] * lmrk_xs,
#                            0.0, config.actual_screen_size[0])
#     pixel_ys = torch.clamp(config.actual_screen_size[1] * lmrk_ys,
#                            0.0, config.actual_screen_size[1])
#     return torch.stack([pixel_xs, pixel_ys], axis=-1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

    
def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    # a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    # b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b
    
    a = pitchyaw_to_vector(a).numpy() 
    b = pitchyaw_to_vector(b).numpy()

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-8, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-8, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    similarity = np.clip(similarity, a_min=-1.+1e-8, a_max=1.-1e-8)
    return np.degrees(np.arccos(similarity))


def angular_error_2(a, b):
    a = pitchyaw_to_vector(a)
    b = pitchyaw_to_vector(b)
    sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
    sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
    return torch.acos(sim) * 180. / np.pi


def compute_angle_error(predictions: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y, pred_z = convert_to_unit_vector(predictions)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


def convert_to_unit_vector(
        angles: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pitches = angles[:, 0]
    yaws = angles[:, 1]
    x = -torch.cos(pitches) * torch.sin(yaws)
    y = -torch.sin(pitches)
    z = -torch.cos(pitches) * torch.cos(yaws)
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def get_target_on_screen(point_on_screen_px, monitor_mm, monitor_pixels, screen_height_mm_offset, screem_depth_mm_offset):
    screen_width_ratio = monitor_mm[0] / monitor_pixels[0]
    screen_height_ratio = monitor_mm[1] / monitor_pixels[1]
    

    # point_on_screen_mm = (monitor_mm[0] / 2 - point_on_screen_px[0] * screen_width_ratio, point_on_screen_px[1] * screen_height_ratio + screen_height_mm_offset)
    point_on_screen_mm = np.array([monitor_mm[0] / 2 - point_on_screen_px[0] * screen_width_ratio, point_on_screen_px[1] * screen_height_ratio + screen_height_mm_offset,  screem_depth_mm_offset])
    return point_on_screen_mm


def vector_to_angle(vector):
    assert vector.shape == (3, )
    vector = vector / np.linalg.norm(vector, axis=0)

    x, y, z = vector
    # pitch = np.arcsin(-y)
    # yaw = np.arctan2(-x, -z)
    
    pitch = np.degrees(np.arcsin(y))
    yaw = np.degrees(np.arctan2(x, z))
    
    return np.array([pitch, yaw])


def angular_error_no_pitchyaw_to_vector(a, b):
    """Calculate angular error (via cosine similarity)."""
    # a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    # b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b
    

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-8, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-8, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    similarity = np.clip(similarity, a_min=-1.+1e-8, a_max=1.-1e-8)
    return np.degrees(np.arccos(similarity))