# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from data import create_dataset
import util.util as util
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transformsn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import sys
from configs.config_test import cfg


if cfg.dataset_mode == 'CIFAR10':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_Size,
                                              shuffle=False, num_workers=2)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif cfg.dataset_mode == 'CIFAR100':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_Size,
                                              shuffle=False, num_workers=2)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif cfg.dataset_mode == 'CelebA':
    dataset = create_dataset(cfg)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
elif cfg.dataset_mode == 'OpenImage':
    dataset = create_dataset(cfg)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
else:
    raise Exception('Not implemented yet')

    
model = create_model(cfg)      # create a model given opt.model and other options
model.setup(cfg)               # regular setup: load and print networks; create schedulers
model.eval()



PSNR_list = []
SSIM_list = []
H_err_MMSE_list = []
H_err_list = []
PAPR_list = []


for i, data in enumerate(dataset):
    if i >= cfg.num_test:  # only apply our model to opt.num_test images.
        break

    start_time = time.time()

    if cfg.dataset_mode in ['CIFAR10', 'CIFAR100']:
        input = data[0]
    elif cfg.dataset_mode in ['CelebA', 'OpenImage']:
        input = data['data']

    model.set_input(input.repeat(cfg.how_many_channel, 1, 1, 1))
    model.forward()
    fake = model.fake
    PAPR = torch.mean(10 * torch.log10(model.PAPR))
    PAPR_list.append(PAPR.item())

    if cfg.feedforward in ['OFDM-CE-sub-EQ', 'OFDM-CE-sub-EQ-sub']:
        H_err_MMSE = torch.mean((model.H_est_MMSE-model.H_true)**2)*2
        H_err = torch.mean((model.H_est-model.H_true)**2)*2
        H_err_MMSE_list.append(H_err_MMSE.item())
        H_err_list.append(H_err.item())
    else:
        H_err_list.append(0)
        H_err_MMSE_list.append(0)

    # Get the int8 generated images
    img_gen_numpy = fake.detach().cpu().float().numpy()
    img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8)

    origin_numpy = input.detach().cpu().float().numpy()
    origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    origin_int8 = origin_numpy.astype(np.uint8)

    diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2, (1, 2, 3))

    PSNR = 10 * np.log10((255**2) / diff)
    PSNR_list.append(np.mean(PSNR))

    img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
    origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

    ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(cfg.how_many_channel, 1, 1, 1), data_range=255, size_average=False)  # return (N,)
    SSIM_list.append(torch.mean(ssim_val).item())

    # Save the first sampled image
    save_path = f'{cfg.image_out_path}/{i}_PSNR_{PSNR[0]:.3f}_SSIM_{ssim_val[0]:.3f}.png'
    util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path, aspect_ratio=1)

    save_path =  f'{cfg.image_out_path}/{i}.png'
    util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)

    if i % 10 == 0:
        print(i)
        

print(f'PSNR: {np.mean(PSNR_list)}')
print(f'SSIM: {np.mean(SSIM_list)}')
print(f'CE refined: {np.mean(H_err_list)}')
print(f'CE MMSE: {np.mean(H_err_MMSE_list)}')
print(f'PAPR: {np.mean(PAPR_list)}')
