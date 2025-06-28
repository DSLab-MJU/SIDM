import os
import re
import numpy as np
import cv2
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interpn
import torch
import torch.nn.functional as F

def interp_shape(top, bottom, num=1):
    """
    Interpolates between two binary masks to generate multiple binary masks.

    Args:
        top (ndarray): Binary mask of the top contour.
        bottom (ndarray): Binary mask of the bottom contour.
        num (int): Number of slices to interpolate. Must be a positive integer.

    Returns:
        ndarray: A 4D array of shape (num, img_size, img_size, 1) containing the interpolated binary masks.
    """
    if num <= 0 or not isinstance(num, int):
        raise ValueError("Number of slices to interpolate must be a positive integer.")
    
    # Compute signed distance maps of the binary masks
    top_dist = signed_bwdist(top)
    bottom_dist = signed_bwdist(bottom)

    # Get mask dimensions
    r, c = top.shape

    # Define grid points for existing data (z=0 is bottom, z=1 is top)
    points = (np.arange(r), np.arange(c), [0, 1])

    # Generate grid for interpolation (including start and end)
    xi = np.linspace(1, 0, num + 2)

    # Create meshgrid for interpolation (r, c, num)
    grid_r, grid_c, grid_z = np.meshgrid(np.arange(r), np.arange(c), xi, indexing='ij')
    grid = np.stack((grid_r, grid_c, grid_z), axis=-1)

    # Stack bottom and top along the z-axis
    data = np.stack((bottom_dist, top_dist), axis=-1)  # Shape: (r, c, 2)

    # Perform interpolation
    out = interpn(points, data, grid, method='linear', bounds_error=False, fill_value=0)

    # Threshold to create binary masks
    out_binary = out >= 0

    # Rearrange axes to (num, r, c) and add channel dimension
    final_masks = np.transpose(out_binary, (2, 0, 1))  # Shape: (num, r, c)
    final_masks = final_masks[..., np.newaxis]         # Shape: (num, r, c, 1)

    # Final mask shape
    return final_masks

def signed_bwdist(im):
    """
    Computes the signed distance map of a binary mask.

    Args:
        im (ndarray): Binary mask.

    Returns:
        ndarray: Signed distance map.
    """
    # Compute distance transforms
    outside_dist = distance_transform_edt(~im)  # Outside distance (negative)
    inside_dist = distance_transform_edt(im)    # Inside distance (positive)

    # Combine to create signed distance map
    signed_dist = -outside_dist * (~im) + inside_dist * im
    return signed_dist


def interpolate_class_pytorch(first_number, second_number, num_classes, num_interpolation):
    """
    Generates interpolated background semantic labels.

    Args:
        first_number (int): The starting class label.
        second_number (int): The ending class label.
        num_classes (int): Total number of classes for one-hot encoding.
        num_interpolation (int): Number of interpolation steps between the start and end labels, excluding the endpoints.
                                 For example, if num_interpolation=3, the interpolation weights will be
                                 [0.0, 0.25, 0.5, 0.75, 1.0], yielding 5 total points.

    Returns:
        Tensor: A tensor of shape (num_interpolation+2, num_classes) containing the interpolated label vectors.
    """
    # Convert the start and end labels to one-hot encoded vectors.
    first_label = F.one_hot(torch.tensor([first_number]), num_classes=num_classes)
    second_label = F.one_hot(torch.tensor([second_number]), num_classes=num_classes)
    first_label = first_label.float()
    second_label = second_label.float()

    # Calculate the interpolation vector between the two labels.
    percent_second_label = torch.linspace(0, 1, (num_interpolation+2))[:, None]
    interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label

    return interpolation_labels

def generate_slerp_noise(x0,x1,num_intervals):
    def slerp(val, low, high):
        low = np.array(low).flatten()
        high = np.array(high).flatten()
        omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
        so = np.sin(omega)
        result = np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high
        return result.reshape(3, 256,256)
    interpolated_images = []
    for i in range(num_intervals + 2):
        t = i / (num_intervals+1)
        print(t)
        interpolated = slerp(t, x0, x1)
        interpolated_images.append(interpolated)

    return np.stack(interpolated_images)
