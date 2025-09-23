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

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > threshold).float()),  # Binarization step
    transforms.Normalize((0.5,), (0.5,))
])

print(len(data), len(mask), len(cond_label), num_labels)
dataset = CustomDataset(data, mask, cond_label, transform=image_transform, mask_transform=mask_transform, num_labels=num_labels)
batch_size = args.batch_size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



diffusion = Diffusion(nn_model=unet, betas=(args.beta1, args.beta2), n_T=args.n_T, device=device, drop_prob=args.dp)
diffusion.to(device)
optim = torch.optim.Adam(diffusion.parameters(), lr=args.lrate)
save_dir = args.save_dir

n_epoch = args.n_epoch
lrate= args.lrate

for ep in range(n_epoch):
    print(f'Epoch {ep+1}')
    diffusion.train()

    # linear lrate decay
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

    pbar = tqdm(dataloader)
    loss_ema = None
    for x, cond in pbar: #cond = [mask, label]
        optim.zero_grad()
        x = x.to(device)
        mask = cond[0].to(device)
        cond_label = cond[1].to(device)
        loss = diffusion(x, [mask, cond_label])
        loss.backward()
        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        optim.step()

    # for eval, save an image of currently generated samples (top rows)
    # followed by real images (bottom rows)
    if ep%50==0:
        torch.save({
                'epoch': ep,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(save_dir+'/output/models/' + f'model_ep{ep}.pth'))
        diffusion.eval()
        with torch.no_grad():
            n_sample = 1
            all_gen_images = []

            for i in range(len(test_masks)):
                x_gen = diffusion.sample(n_sample, (3,256,256), device, condition=[test_masks[i].unsqueeze(0),test_labels[i].unsqueeze(0)] , guide_w=1.5) 
                x_gen = (x_gen + 1) / 2
                x_gen = x_gen.clamp(0,1)
                all_gen_images.append(x_gen)
            all_images = torch.cat(all_gen_images, dim=0)

            img_grid = make_grid(all_images, nrow=5)
            plt.figure(figsize=(12,8))
            plt.imshow(np.transpose(img_grid.cpu().numpy(), (1, 2, 0)))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_dir + '/output/imgs/'  + f'sampleimgs_ep{ep}.png', bbox_inches='tight', pad_inches=0.0)
            plt.show()
            plt.close()