import numpy as np
import time
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
import os
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
import traceback
import torchvision
from torchvision import transforms
import random
import shutil

from kornia import create_meshgrid

import pdb

def generate_plucker_rays(T, shape, fov, sensor_size=(1.0, 1.0)):
    """
    Generate Plücker rays for each pixel of the image based on a camera transformation matrix.

    Args:
    T (numpy.ndarray): A 4x4 transformation matrix representing the camera pose.
    H (int): Height of the image.
    W (int): Width of the image.
    focal_length_x (float): The focal length of the camera in the X direction.
    focal_length_y (float): The focal length of the camera in the Y direction.
    sensor_size (tuple): Physical size of the sensor in world units (width, height).

    Returns:
    numpy.ndarray: A (6, H, W) array where the first 3 elements are the direction vector (d)
                   and the last 3 are the moment vector (m) for each pixel.
    """
    # Extract the rotation matrix (3x3) and translation vector (3x1) from the transformation matrix
    R = T[:3, :3]  # Rotation part
    t = T[:3, 3]   # Translation part (camera position in world space)
    H, W = shape
    H //= 8
    W //= 8
    focal_length_x, focal_length_y = fov

    # Generate pixel grid in image space
    i = np.linspace(-sensor_size[1] / 2, sensor_size[1] / 2, H)  # Y coordinates
    j = np.linspace(-sensor_size[0] / 2, sensor_size[0] / 2, W)  # X coordinates

    # Create 2D meshgrid for pixel centers
    J, I = np.meshgrid(j, i)

    # Compute normalized camera rays (camera space, assuming pinhole camera model)
    rays_d_cam = np.stack([
        J / focal_length_x,  # Scale by focal length in X
        I / focal_length_y,  # Scale by focal length in Y
        np.full_like(J, 1.0)  # Z is fixed to 1.0 for normalized camera rays
    ], axis=-1)  # Shape: (H, W, 3)

    # Normalize ray directions
    rays_d_cam /= np.linalg.norm(rays_d_cam, axis=-1, keepdims=True)  # Normalize to unit vectors

    # Transform ray directions to world space using the rotation matrix
    rays_d_world = -np.matmul(R, rays_d_cam.reshape(-1, 3).T).T.reshape(H, W, 3)  # Shape: (H, W, 3)

    # Moment vector for each pixel is computed as t x d (cross product of translation and direction)
    rays_m_world = np.cross(t, rays_d_world, axisa=0, axisb=2)  # Shape: (H, W, 3)

    # Combine direction vectors and moment vectors into a single array of shape (6, H, W)
    plucker_rays = np.stack([
        rays_d_world[..., 0], rays_d_world[..., 1], rays_d_world[..., 2],
        rays_m_world[..., 0], rays_m_world[..., 1], rays_m_world[..., 2]
    ], axis=0)
    # origins = np.tile(t[:, np.newaxis, np.newaxis], (1, H, W))  # Shape: (3, H, W)
    return plucker_rays

##### Below are functions for preprocessing environment map, copied from Neural Gaffer #####

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def read_hdr(path):
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    try:
        with open(path, 'rb') as h:
            buffer_ = np.frombuffer(h.read(), np.uint8)
        bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # (1024, 2048, 3)
        
    except Exception as e:
        print(f"Error reading HDR file {path}: {e}")
        rgb = None
        return rgb
    rgb = torch.from_numpy(rgb)
    return rgb

def generate_envir_map_dir(envmap_h, envmap_w):
    lat_step_size = np.pi / envmap_h
    lng_step_size = 2 * np.pi / envmap_w
    theta, phi = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h), 
                                torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)], indexing='ij')

    sin_theta = torch.sin(torch.pi / 2 - theta)  # [envH, envW]
    light_area_weight = 4 * torch.pi * sin_theta / torch.sum(sin_theta)  # [envH, envW]
    assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
    light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]


    view_dirs = torch.stack([   torch.cos(phi) * torch.cos(theta), 
                                torch.sin(phi) * torch.cos(theta), 
                                torch.sin(theta)], dim=-1).view(-1, 3)    # [envH * envW, 3]
    light_area_weight = light_area_weight.reshape(envmap_h, envmap_w)
    
    return light_area_weight, view_dirs

def get_light(hdr_rgb, incident_dir, hdr_weight=None):

    envir_map = hdr_rgb
    envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
    if torch.isnan(envir_map).any():
        os.system('echo "nan in envir_map"')
    if hdr_weight is not None:
        hdr_weight = hdr_weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
    incident_dir = incident_dir.clip(-1, 1)
    theta = torch.arccos(incident_dir[:, 2]).reshape(-1) - 1e-6 # top to bottom: 0 to pi
    phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1) # left to right: pi to -pi

    #  x = -1, y = -1 is the left-top pixel of F.grid_sample's input
    query_y = (theta / np.pi) * 2 - 1 # top to bottom: -1-> 1
    query_y = query_y.clip(-1, 1)
    query_x = - phi / np.pi # left to right: -1 -> 1
    query_x = query_x.clip(-1, 1)
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float() # [1, 1, 2, N]
    if abs(grid.max()) > 1 or abs(grid.min()) > 1:
        os.system('echo "grid out of range"')
    
    light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

    if torch.isnan(light_rgbs).any():
        os.system('echo "nan in light_rgbs"')
    return light_rgbs    


def process_im(im):
    im = im.convert("RGB")
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256), antialias=True),  # 256, 256
            transforms.ToTensor(), # for PIL to Tensor [0, 255] -> [0.0, 1.0] and H×W×C-> C×H×W
            transforms.Normalize([0.5], [0.5]) # x -> (x - 0.5) / 0.5 == 2 * x - 1.0; [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
    return image_transforms(im)



def rotate_and_preprcess_envir_map(envir_map, aligned_RT, rotation_idx=0, total_view=120):
    # envir_map: [H, W, 3]
    # aligned_RT: numpy.narray [3, 4] w2c
    # the coordinate system follows Blender's convention
    
    # c_x_axis, c_y_axis, c_z_axis = aligned_RT[0, :3], aligned_RT[1, :3], aligned_RT[2, :3]
    env_h, env_w = envir_map.shape[0], envir_map.shape[1]
 
    light_area_weight, view_dirs = generate_envir_map_dir(env_h, env_w)
    
    axis_aligned_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # Blender's convention
    axis_aligned_R = axis_aligned_transform @ aligned_RT[:3, :3] # [3, 3]
    view_dirs_world = view_dirs @ axis_aligned_R # [envH * envW, 3]
    
    # rotate the envir map along the z-axis
    rotated_z_radius = (-2 * np.pi * rotation_idx / total_view) 
    # [3, 3], left multiplied by the view_dirs_world
    rotation_maxtrix = np.array([[np.cos(rotated_z_radius), -np.sin(rotated_z_radius), 0],
                                [np.sin(rotated_z_radius), np.cos(rotated_z_radius), 0],
                                [0, 0, 1]])
    view_dirs_world = view_dirs_world @ rotation_maxtrix        
    
    rotated_hdr_rgb = get_light(envir_map, view_dirs_world)
    rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3)
    
    rotated_hdr_rgb = np.array(rotated_hdr_rgb, dtype=np.float32)

    # ldr
    envir_map_ldr = rotated_hdr_rgb.clip(0, 1)
    envir_map_ldr = envir_map_ldr ** (1/2.2)
    # hdr
    envir_map_hdr = np.log1p(10 * rotated_hdr_rgb)
    # rescale to [0, 1]
    envir_map_hdr = envir_map_hdr / np.max(envir_map_hdr)
    envir_map_ldr = np.uint8(envir_map_ldr * 255)
    envir_map_ldr = Image.fromarray(envir_map_ldr)
    envir_map_hdr = np.uint8(envir_map_hdr * 255)
    envir_map_hdr = Image.fromarray(envir_map_hdr)

    return envir_map_ldr, envir_map_hdr

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5 # 1xHxWx2

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions

def get_ray_d(input_RT):
    sensor_width = 32

    # Get camera focal length
    focal_length = 35

    # Get image resolution
    resolution_x = 256

    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Compute focal length in pixels
    focal_length_px_x = focal_length * resolution_x / sensor_width

    focal = focal_length_px_x
    
    directions = get_ray_directions(resolution_x, resolution_x, [focal, focal])  # [H, W, 3]
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    
    w2c = input_RT
    w2c = np.vstack([w2c, [0, 0, 0, 1]])  # [4, 4]
    c2w = np.linalg.inv(w2c)
    pose = c2w @ blender2opencv
    c2w = torch.FloatTensor(pose)  # [4, 4]
    w2c = torch.linalg.inv(c2w)  # [4, 4]
    # Read ray data
    _, rays_d = get_rays(directions, c2w)

    return rays_d

def get_envir_map_light(envir_map, incident_dir):

    envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
    phi = torch.arccos(incident_dir[:, 2]).reshape(-1) - 1e-6
    theta = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1)
    # normalize to [-1, 1]
    query_y = (phi / np.pi) * 2 - 1
    query_x = - theta / np.pi
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
    light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

    return light_rgbs

def _clip_0to1_warn_torch(tensor_0to1):
    """Enforces [0, 1] on a tensor/array that should be already [0, 1].
    """
    msg = "Some values outside [0, 1], so clipping happened"
    if isinstance(tensor_0to1, torch.Tensor):
        if torch.min(tensor_0to1) < 0 or torch.max(tensor_0to1) > 1:
            tensor_0to1 = torch.clamp(
                tensor_0to1, min=0, max=1)
    elif isinstance(tensor_0to1, np.ndarray):
        if tensor_0to1.min() < 0 or tensor_0to1.max() > 1:
            tensor_0to1 = np.clip(tensor_0to1, 0, 1)
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')
    return tensor_0to1

def linear2srgb_torch(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    elif isinstance(tensor_0to1, np.ndarray):
        pow_func = np.power
        where_func = np.where
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = _clip_0to1_warn_torch(tensor_0to1)

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1 + 1e-6, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


def split_dataset(source_dir, train_dir, val_dir, val_ratio=0.05):
    # Create train and val directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all HDR files from the source directory
    hdri_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    # Calculate number of files for validation set
    num_files = len(hdri_folders)
    num_val = int(num_files * val_ratio)
    
    # Randomly select files for validation set
    val_files = random.sample(hdri_folders, num_val)
    
    # Copy files to respective directories
    for folder in hdri_folders:
        src_folder = os.path.join(source_dir, folder)
        
        if folder in val_files:
            dst_folder = os.path.join(val_dir, folder)
        else:
            dst_folder = os.path.join(train_dir, folder)
            
        shutil.copytree(src_folder, dst_folder)
    
    # Print statistics
    print(f"Total files: {num_files}")
    print(f"Training files: {num_files - num_val}")
    print(f"Validation files: {num_val}")

if __name__ == "__main__":
    # Define paths
    source_path = "../datasets/haven/hdris"
    train_path = "../datasets/haven/hdris/train"
    val_path = "../datasets/haven/hdris/val"
    
    random.seed(42)
    
    split_dataset(source_path, train_path, val_path)
