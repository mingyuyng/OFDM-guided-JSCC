# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from models import create_model
from data import create_dataset
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from config import cfg

cfg.isTrain = True

# Create dataloaders
if cfg.dataset_mode == 'CIFAR10':
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomCrop(cfg.sizew, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=cfg.dataroot, train=True,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif cfg.dataset_mode == 'CIFAR100':
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomCrop(cfg.sizew, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root=cfg.dataroot, train=True,
                                             download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=cfg.batchsize,
                                          shuffle=True, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif cfg.dataset_mode == 'CelebA':
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif cfg.dataset_mode == 'OpenImage':
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
else:
    raise Exception('Not implemented yet')


model = create_model(cfg)      # create a model given cfg.model and other options
model.setup(cfg)               # regular setup: load and print networks; create schedulers
visualizer = Visualizer(cfg)   # create a visualizer that display/save images and plots
total_iters = 0                # the total number of training iterations

# Train with the Discriminator
for epoch in range(cfg.epoch_count, cfg.n_epochs + cfg.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % cfg.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += 1
        epoch_iter += 1

        if cfg.dataset_mode in ['CIFAR10', 'CIFAR100']:
            input = data[0]
        elif cfg.dataset_mode == 'CelebA':
            input = data['data']
        elif cfg.dataset_mode == 'OpenImage':
            input = data['data']

        model.set_input(input)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        if total_iters % cfg.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % cfg.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_iters % cfg.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time)
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if cfg.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if total_iters % cfg.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if cfg.save_by_iter else 'latest'
            model.save_networks(save_suffix)
        iter_data_time = time.time()

    if epoch % cfg.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, cfg.n_epochs + cfg.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
