import numpy as np
import torch


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