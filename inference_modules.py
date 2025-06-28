import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random

from PIL import Image
import blobfile as bf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Compose
from torchvision import transforms
import cv2

import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import torch.nn as nn
import pandas as pd

def visualize_masks_and_images(masks, images):
    masks = np.array(masks)
    images = np.array(images)
    
    if masks.shape[0] != images.shape[0]:
        raise ValueError("Mask & Image NumError")
    
    num = masks.shape[0]
    
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks.squeeze(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)

    if images.ndim == 4 and images.shape[-1] in [1, 3]:
        pass  
    elif images.ndim == 3:
        pass  
    else:
        raise ValueError("Image's shape Error")
    
    fig, axes = plt.subplots(2, num, figsize=(num * 2, 4))
    
    for i in range(num):
        # First Row: Masks
        ax = axes[0, i]
        ax.imshow(masks[i], cmap='gray')
        ax.axis('off')
        title = "Ground Truth" if (i == 0 or i == num-1) else f"Mask {i}"
        ax.set_title(title)
        
        # Second Row: Images
        ax = axes[1, i]
        if images.ndim == 4 and images.shape[-1] == 3:
            ax.imshow(images[i])
        elif images.ndim == 4 and images.shape[-1] == 1:
            ax.imshow(images[i].squeeze(-1), cmap='gray')
        else:
            ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        title = "Ground Truth" if (i == 0 or i == num-1) else f"Image {i}"
        ax.set_title(title)

# Visualize the results in a single row
def visualize_slices_in_row(interpolated_masks):
    """
    Visualize all interpolated masks in a single row.

    Args:
        interpolated_masks (ndarray): 3D array of interpolated masks (r, c, num_slices).

    Returns:
        None
    """
    num_slices = interpolated_masks.shape[0]  # Number of slices
    plt.figure(figsize=(15, 5))  # Adjust the figure size based on number of slices
    
    for i in range(num_slices):
        plt.subplot(1, num_slices, i + 1)  # 1 row, num_slices columns
        plt.imshow(interpolated_masks[i].reshape(interpolated_masks.shape[1],interpolated_masks.shape[2]), cmap='gray')
        plt.title(f"Slice {i+1}")
        plt.axis('off')  # Hide axes for better visualization

    plt.tight_layout()
    plt.show()

import blobfile as bf

def list_image_files_recursively(data_dir, key=None):
    results = []
    for entry in sorted(bf.listdir(data_dir), key=key):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[0])
    return 0


def process_directory_fortestmask(image_path):
    image = cv2.resize(cv2.imread(image_path), (256, 256))[:, :, 1]
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_array = np.array(image)
    return image_array

mask_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def interpolated_mask_preprocess(interpolated_masks):
    test_mask = []
    for i in range(len(interpolated_masks)):
        tensor = mask_preprocess(interpolated_masks[i].astype(float))
        test_mask.append(tensor)
        
    test_masks = torch.stack(test_mask) 
    return test_masks