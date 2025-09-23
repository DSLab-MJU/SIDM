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
from interpolations import *
from params import *

args = parse_arguments()
data_dir = args.data_path

def _list_image_files_recursively(data_dir, key=None):
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
        return int(numbers[-1])
    return 0

all_files = _list_image_files_recursively(os.path.join(data_dir, 'images'))
all_files_video = _list_image_files_recursively(os.path.join(data_dir, 'images_video'), key = extract_number)
classes = _list_image_files_recursively(os.path.join(data_dir, 'masks'))
classes_video = _list_image_files_recursively(os.path.join(data_dir, 'masks_video'), key = extract_number)
instances = None

print(len(all_files), len(all_files_video), len(classes), len(classes_video))

print(all_files[:5], classes[:5])
print(all_files_video[:5], classes_video[:5])


# Creating a dictionary to map frame ranges to sequence numbers
sequence_mapping = {
    (1, 25): 1, (26, 50): 2, (51, 67): 3, (68, 78): 4, (79, 103): 5, (104, 126): 6, (127, 151): 7, (152, 177): 8,
    (178, 199): 9, (200, 205): 10, (206, 227): 11, (228, 252): 12, (253, 277): 13, (278, 297): 14, (298, 317): 15, (318, 342): 16,
    (343, 363): 17, (364, 383): 18, (384, 408): 19, (409, 428): 20, (429, 447): 21, (448, 466): 22, (467, 478): 23, (479, 503): 24,
    (504, 528): 25, (529, 546): 26, (547, 571): 27, (572, 591): 28, (592, 612): 29
}

# Expanding the dictionary to map individual frame numbers to sequences
frame_to_sequence = {}
for (start, end), sequence in sequence_mapping.items():
    for frame in range(start, end + 1):
        frame_to_sequence[frame] = sequence

# Convert to DataFrame for better visualization
df = pd.DataFrame(frame_to_sequence.items(), columns=["Frame", "Sequence"])

data = []
mask = []
cond_label = []

img_size = 256

# Transform for the input images
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor() 
])

# Define the threshold for binarization
threshold = 0.5  # Normalized threshold for ToTensor() output (range: 0 to 1)

# Transform for the mask images with binarization
mask_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),  # Convert to tensor (values scaled to [0,1])
    transforms.Lambda(lambda x: (x > threshold).float())  # Binarization step
])


for i in range(len(all_files_video)):
    # Process the input image
    #print(all_files[i])
    image = Image.open(all_files_video[i])
    image = transform(image)
    data.append(image)

    # Process the mask image
    mask_image = Image.open(classes_video[i])
    #print(classes[i])
    mask_image = mask_transform(mask_image)
    mask.append(mask_image)

    # Extract the last filename after splitting by "/"
    video_filename = str(all_files_video[i]).split("/")[-1]
    mask_filename = str(classes_video[i]).split("/")[-1]
    # Check if filenames match, and issue a warning if they don't
    if video_filename != mask_filename:
        print(f"⚠ WARNING: Mismatch detected - Video: {video_filename}, Mask: {mask_filename}")

    # Remove the ".png" extension and convert to an integer
    last_value = int(os.path.splitext(mask_filename)[0])  # Extract "1" and convert to int

    # Assign label based on `frame_to_sequence` mapping
    sequence_label = frame_to_sequence.get(last_value, None)
    print(all_files_video[i], classes_video[i],sequence_label)

    if sequence_label:
        cond_label.append(sequence_label)
    #print()

'''
 Dataset info:
    According to the CVC-ClinicDB (https://polyp.grand-challenge.org/CVCClinicDB/),
    612 frames were extracted from 29 videos.
'''    
cnt = 30 # According to CVC-ClinicDB, background semantic label begins at 30.
for i in range(len(all_files)):
    # Process the input image
    image = Image.open(all_files[i])
    image = transform(image)
    data.append(image)

    # Process the mask image
    mask_image = Image.open(classes[i])
    mask_image = mask_transform(mask_image)
    mask.append(mask_image)

    # Extract the last filename after splitting by "/"
    video_filename = str(all_files[i]).split("/")[-1]
    mask_filename = str(classes[i]).split("/")[-1]
    # Check if filenames match, and issue a warning if they don't
    if video_filename != mask_filename:
        print(f"⚠ WARNING: Mismatch detected - Video: {video_filename}, Mask: {mask_filename}")

    sequence_label = cnt
    print(all_files[i], classes[i],sequence_label)

    if sequence_label:
        cond_label.append(sequence_label)
    cnt +=1 

num_labels = cnt

class CustomDataset(Dataset):
    def __init__(self, images, masks, cond_labels, transform=None, mask_transform=None, num_labels=None):
        self.images = images
        self.masks = masks
        self.cond_labels = cond_labels
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_labels = num_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        cond_label = self.cond_labels[idx]

        # 이미지와 레이블을 PIL 이미지로 변환
        image = ToPILImage()(image)
        mask = ToPILImage()(mask)

        # 이미지에 대한 변환 적용
        if self.transform:
            image = self.transform(image)

        # 레이블(마스크)에 대한 변환 적용
        if self.mask_transform:
            mask = self.mask_transform(mask)

        cond_label = nn.functional.one_hot(torch.tensor(cond_label), num_classes=num_labels).type(torch.float) 

        return image, [mask,cond_label]




'''
!!!!!!!!!!! for eval during Training !!!!!!!!!!!
'''
mask_data = []
mask_file_name = []
def process_directory_fortestmask(directory_path):
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            image = cv2.resize(cv2.imread(image_path), (256, 256))[:, :, 1]
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_array = np.array(image)
            mask_data.append(image_array)
            mask_file_name.append(filename)

directory_path = args.data_path+'/masks/'
process_directory_fortestmask(directory_path)

mask1_idx = 0 #label=30
mask2_idx = 6 #label=36 
mask1 = mask_data[mask1_idx]
#print(mask_file_name[mask1_idx])
mask2 = mask_data[mask2_idx]
#print(mask_file_name[mask2_idx])
# Interpolate between mask1 and mask2 with 3 slices
num_slices = 3
interpolated_masks = interp_shape(mask1, mask2, num_slices)


import cv2
from torchvision import transforms
import torch

mask_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_mask = []
for i in range(len(interpolated_masks)):
    if interpolated_masks[i].shape == (args.image_size,args.image_size,1):
        pass
    else:
        print('Interpoalted Mask Shape Error')
    tensor = mask_preprocess(interpolated_masks[i].astype(float))
    test_mask.append(tensor)
test_masks = torch.stack(test_mask) 
#print(test_masks.shape)

test_labels = interpolate_class_pytorch(30,36,num_labels,3)
#print(test_labels, test_labels.shape)

#print(len(test_masks))
#print(test_masks[i].shape)