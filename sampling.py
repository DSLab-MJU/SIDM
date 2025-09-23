import torch
print(torch.cuda.is_available())

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

print(torch.cuda.get_device_name()) 

print(torch.cuda.device_count())

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import re

from params import *

from interpolations import *
from modules import *
from polyp import *
from U_Net_label import *
from diffusion import *

args = parse_arguments()

unet = create_model(
        args.image_size, 
        args.num_classes,
        args.num_channels,
        num_labels, 
        args.num_res_blocks, 
        channel_mult=args.channel_mult, 
        learn_sigma=args.learn_sigma, 
        class_cond=args.class_cond, 
        use_checkpoint=args.use_checkpoint, 
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        resblock_updown=args.resblock_updown,
        use_fp16=args.use_fp16,
        use_new_attention_order=args.use_new_attention_order,
        no_instance=args.no_instance,
    )


# Function to extract the last number from a filename using regex
def extract_number(filename):
    # Use a regex pattern to find numeric parts in the filename. \d+ matches one or more digits.
    numbers = re.findall(r"\d+", filename)
    # If any numbers are found, return the last one as an integer.
    if numbers:
        return int(numbers[-1])
    # If no numbers are found, return 0.
    return 0

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
'''
Selecting Pair algorithm
1. Pairs with different background semantic labels were considered distinct, regardless of mask similarity. 
2. For masks, selection was based on size comparison with a threshold at zero.
'''
interpol_masks_list = []
interpol_labels_list = []

# Number of interpolation segments. For example, ratios 1:3, 1:1, 3:1 correspond to num_slices=3
num_slices = 3 

# Loop over masks to generate interpolations
for image_idx in range(len(masks)):
    interpol_masks = []
    # Stop before the last index to avoid out-of-range
    if image_idx == len(masks)-1: break 
    idx = 1
    print(f'# index: {image_idx}')
    while True:
        # Select two different masks
        image1 = masks[image_idx]
        image2 = masks[image_idx+idx]   
        
        interpolated_masks = interp_shape(image1, image2, num_slices)
        print(np.unique(interpolated_masks[1]), np.unique(interpolated_masks[2]), np.unique(interpolated_masks[3]))

        print(image_idx, image_idx+idx ,'/', cond_label[image_idx], cond_label[image_idx+idx])
        
        # After interpolating two masks, if any interpolated mask's size is zero(threshold), select the next mask
        # you can change the threshold for your setting
        if np.all(np.unique(interpolated_masks[1]) == False) or np.all(np.unique(interpolated_masks[2]) == False) or np.all(np.unique(interpolated_masks[3]) == False): 
            idx += 1
            continue
        break

    # Generate interpolated class labels between the two conditions
    interpol_labels = interpolate_class_pytorch(cond_label[image_idx], cond_label[image_idx+idx],num_labels,num_slices)
    interpol_labels_list.append(interpol_labels)
    
    if len(interpolated_masks) != (num_slices+2) :
        print('Num Slices Error!!')        

    # Preprocess and stack interpolated masks
    for i in range(len(interpolated_masks)):
        if interpolated_masks[i].shape == (256,256,1):
            pass
        else:
            print('Interpoalted Mask Shape Error')
        interpol_mask = mask_preprocess(interpolated_masks[i].astype(float))
        interpol_masks.append(interpol_mask)
    interpol_masks = torch.stack(interpol_masks) 
    interpol_masks_list.append(interpol_masks)

# Initialize pre-trained diffusion model and optimizer
diffusion = Diffusion(nn_model=unet, betas=(args.beta1, args.beta2), n_T=args.n_T, device=device, drop_prob=args.dp)
optim = torch.optim.Adam(diffusion.parameters(), lr=args.lrate)

# Load checkpoint
save_dir = args.save_dir
checkpoint = torch.load(save_dir+f' ', map_location=device)  #your trained model!
diffusion.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

'''
- Example -
Select the second slice (1:1 ratio) from the interpolated segments.
Thus, inter_step=2 with num_slices=3 corresponds to the middle slice of the 1:1 split. 
'''
inter_step = 2 #!!!!!Input!!!!!
diffusion.eval()
with torch.no_grad():
    print(len(interpol_masks_list), len(interpol_labels_list))
    for i in range(len(interpol_masks_list)):
        print(interpol_labels_list[i][inter_step])
        n_sample = 1
        mask_cond = torch.tensor(interpol_masks_list[i][inter_step]) #real_masks[i])
        label_cond = torch.tensor(interpol_labels_list[i][inter_step])

        mask = mask_cond.cpu().numpy().transpose(1,2,0).squeeze()
        mask = (mask + 1) / 2 
        plt.imsave(os.path.join(save_dir+f'/sampled_masks/gen_interstep{inter_step}_mask/', f'interp_mask_{i+1}.png'), mask, cmap='gray')
    
        x_gen = diffusion.sample(n_sample, (3,256,256), device, condition=[mask_cond.unsqueeze(0),label_cond.unsqueeze(0)], guide_w=1.5)
        x_gen = (x_gen + 1) / 2
        x_gen = x_gen.clamp(0,1)

        img = x_gen[0].cpu().numpy().transpose(1, 2, 0).squeeze()
        plt.imsave(os.path.join(save_dir+f'/sampled_imgs/gen_interstep{inter_step}_img/', f'gen_img_{i+1}.png'), img)
        plt.close()