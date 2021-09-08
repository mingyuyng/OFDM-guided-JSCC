# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import channel
import scipy.io as sio
import random

class JSCCOFDMModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L2', 'G_H', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.opt.gan_mode != 'none' and self.isTrain:
            self.model_names = ['E', 'G', 'D']
        else:  # during test time, only load G
            self.model_names = ['E', 'G']
        
        if self.opt.feedforward == 'OFDM-CE-sub-EQ-sub':
            self.model_names += ['CE', 'EQ']
            self.netCE = networks.define_S(dim=6, dim_out=2, dim_in = 32,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.netEQ = networks.define_S(dim=6, dim_out=2, dim_in = 32,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
        elif self.opt.feedforward == 'OFDM-CE-sub-EQ':
            self.model_names += ['CE']
            self.netCE = networks.define_S(dim=6, dim_out=2, dim_in = 32,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
        elif self.opt.feedforward == 'OFDM-feedback':
            self.model_names += ['EQ', 'P']
            self.netEQ = networks.define_S(dim=6, dim_out=2, dim_in = 32,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.netP = networks.define_S(dim=self.opt.C_channel+2, dim_out=self.opt.C_channel, dim_in = 64,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
            
        # define networks (both generator and discriminator)
        self.netE = networks.define_E(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=opt.C_channel,
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type='kaiming',
                                      init_gain=0.02, gpu_ids=self.gpu_ids)

        self.netG = networks.define_G(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                      n_downsample=opt.n_downsample, C_channel=opt.C_channel,
                                      n_blocks=opt.n_blocks, norm=opt.norm_EG, init_type='kaiming',
                                      init_gain=0.02, gpu_ids=self.gpu_ids)

        #if self.isTrain and self.is_GAN:  # define a discriminator;
        if self.opt.gan_mode != 'none':
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D,
                                          opt.norm_D, 'kaiming', 0.02, self.gpu_ids)


        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL2 = torch.nn.MSELoss()

            params = list(self.netE.parameters()) + list(self.netG.parameters())

            if self.opt.feedforward == 'OFDM-CE-sub-EQ':
                params+= list(self.netCE.parameters())
            elif self.opt.feedforward == 'OFDM-CE-sub-EQ-sub':
                params+= list(self.netCE.parameters())
                params+= list(self.netEQ.parameters())
            elif self.opt.feedforward == 'OFDM-feedback':
                params+= list(self.netEQ.parameters())
                params+= list(self.netP.parameters())
                

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if self.opt.gan_mode != 'none':
                params = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
        
        self.opt = opt
        self.channel = channel.OFDM_channel(opt, self.device, pwr=1)


    def name(self):
        return 'JSCCOFDM_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_decode(self, latent):
        self.latent = self.normalize(latent.to(self.device),1)

    def set_img_path(self, path):
        self.image_paths = path

    def forward(self, cof_in=None):

        N = self.real_A.shape[0]

        # Pass the image through the image encoder
        latent = self.netE(self.real_A)

        # Generate information about the channel when available
        if cof_in is not None:
            cof, H_true = cof_in
        elif cof_in is None and self.opt.feedforward == 'OFDM-feedback':
            cof, H_true = self.channel.sample(N)
        else:
            cof, H_true = None, None
        
        # Pre-coding process when the channel feedback is available
        if self.opt.feedforward == 'OFDM-feedback':
            H_true = H_true.permute(0, 1, 3, 2).contiguous().view(N, -1, latent.shape[2], latent.shape[3]).to(latent.device)
            weights = self.netP(torch.cat((H_true, latent), 1))
            latent = latent*weights
        
        # Reshape the latents to be transmitted
        self.tx = latent.view(N, self.opt.P, self.opt.S, 2, self.opt.M).permute(0,1,2,4,3)
        
        # Transmit through the channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR = self.channel(self.tx, SNR=self.opt.SNR, cof=cof)
        self.H_true = self.H_true.to(self.device).unsqueeze(2)

        N, C, H, W = latent.shape

        # The receiver contains CE and EQ modules but no additional subnets
        if self.opt.feedforward == 'OFDM-CE-EQ':
            self.H_est_MMSE = self.channel_estimation(out_pilot, noise_pwr)
            self.H_est = self.H_est_MMSE
            rx = self.equalization(self.H_est, out_sig, noise_pwr)
            dec_in = rx.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W)
            self.fake = self.netG(dec_in)

        # The receiver contains CE and EQ modules plus the CE subnet
        elif self.opt.feedforward == 'OFDM-CE-sub-EQ':
            self.H_est_MMSE = self.channel_estimation(out_pilot, noise_pwr)
            sub11 = self.channel.pilot.repeat(N,1,1,1,1)
            sub12 = torch.mean(out_pilot, 2, True)
            sub13 = self.H_est_MMSE
            sub1_input = torch.cat((sub11, sub12, sub13), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, 8, 8)
            sub1_output = self.netCE(sub1_input).view(N, self.opt.P, 1, 2, self.opt.M).permute(0,1,2,4,3)

            self.H_est = self.H_est_MMSE + sub1_output
            sub21 = self.H_est
            sub22 = out_sig
            self.rx = self.equalization(sub21, sub22, noise_pwr)
            sub23 = self.rx

            dec_in = (self.rx).permute(0,1,2,4,3).contiguous().view(latent.shape)
            self.fake = self.netG(dec_in)
        
        # The receiver contains CE and EQ modules plus the CE and EQ subnet
        elif self.opt.feedforward == 'OFDM-CE-sub-EQ-sub':
            self.H_est_MMSE = self.channel_estimation(out_pilot, noise_pwr)            
            sub11 = self.channel.pilot.repeat(N,1,1,1,1)
            sub12 = torch.mean(out_pilot, 2, True)
            sub13 = self.H_est_MMSE
            sub1_input = torch.cat((sub11, sub12, sub13), 2).contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, 8, 8)
            sub1_output = self.netCE(sub1_input).view(N, self.opt.P, 1, 2, self.opt.M).permute(0,1,2,4,3)

            self.H_est = self.H_est_MMSE + sub1_output
            sub21 = self.H_est
            sub22 = out_sig
            self.rx = self.equalization(sub21, sub22, noise_pwr)
            sub23 = self.rx

            sub2_input = torch.cat((sub21.repeat(1,1,self.opt.S,1,1), sub22, sub23), 3).reshape(-1, 1, 3, self.opt.M, 2).contiguous().permute(0,1,2,4,3).contiguous().view(-1, 6, 8, 8)
            sub2_output = self.netEQ(sub2_input).view(-1, 1, 1, 2, self.opt.M).permute(0,1,2,4,3).view(self.rx.shape)
            dec_in = (self.rx+sub2_output).permute(0,1,2,4,3).contiguous().view(latent.shape)
            self.fake = self.netG(dec_in)
        
        # The case when channel feedback is available. CE module is not needed    
        elif self.opt.feedforward == 'OFDM-feedback':
            self.H_est = self.H_true
            sub21 = self.H_est
            sub22 = out_sig
            self.rx = self.equalization(sub21, sub22, noise_pwr)
            sub23 = self.rx

            sub2_input = torch.cat((sub21.repeat(1,1,self.opt.S,1,1), sub22, sub23), 3).reshape(-1, 1, 3, self.opt.M, 2).contiguous().permute(0,1,2,4,3).contiguous().view(-1, 6, 8, 8)
            sub2_output = self.netEQ(sub2_input).view(-1, 1, 1, 2, self.opt.M).permute(0,1,2,4,3).view(self.rx.shape)
            dec_in = (self.rx+sub2_output).permute(0,1,2,4,3).contiguous().view(latent.shape)
            self.fake = self.netG(dec_in)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        _, pred_fake = self.netD(self.fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_data = self.real_B
        _, pred_real = self.netD(real_data)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        if self.opt.gan_mode in ['lsgan', 'vanilla']:
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        elif self.opt.gan_mode == 'wgangp':
            penalty, grad = networks.cal_gradient_penalty(self.netD, real_data, self.fake.detach(), self.device, type='mixed', constant=1.0, lambda_gp=10.0)
            self.loss_D = self.loss_D_fake + self.loss_D_real + penalty
            self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        if self.opt.gan_mode != 'none':
            feat_fake, pred_fake = self.netD(self.fake)
            self.loss_G_GAN = self.opt.lam_G * self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) * self.opt.lam_L2

        if self.opt.feedforward in ['OFDM-CE-sub-EQ', 'OFDM-CE-sub-EQ-sub']:
            self.loss_G_H = self.opt.lam_h * torch.mean((self.H_est-self.H_true)**2)*2  # Channel estimations are complex numbers
        else:
            self.loss_G_H = 0
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.loss_G_H 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.opt.gan_mode != 'none':
            self.set_requires_grad(self.netD, True)   # enable backprop for D
            self.optimizer_D.zero_grad()              # set D's gradients to zero
            self.backward_D()                         # calculate gradients for D
            self.optimizer_D.step()                   # update D's weights
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        else:
            self.loss_D_fake = 0
            self.loss_D_real = 0
        # update G

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def get_channel(self):
        cof, _ = self.channel.sample()
        return cof


    def channel_estimation(self, out_pilot, noise_pwr):
        return channel.LMMSE_channel_est(self.channel.pilot, out_pilot, self.opt.M*noise_pwr)

    def equalization(self, H_est, out_sig, noise_pwr):
        return channel.MMSE_equalization(H_est, out_sig, self.opt.M*noise_pwr)
        

