from easydict import EasyDict as edict
from configs.config import cfg
import os
import shutil

__E                                              = cfg

# Model config for different datasets
if __E.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __E.batch_size                               = 1
    __E.size_w                                   = 32
    __E.size_h                                   = 32 

elif __E.dataset_mode == 'CelebA':
    __E.batch_size                               = 1
    __E.dataroot                                 = './data/celeba/CelebA_test'
    __E.size_w                                   = 64
    __E.size_h                                   = 64  

elif __E.dataset_mode == 'OpenImage':
    __E.batch_size                               = 1
    __E.dataroot                                 = './data/Kodak'
    __E.size_w                                   = 512
    __E.size_h                                   = 768                                  

__E.verbose                                      = False
__E.serial_batches                               = True
__E.isTrain                                      = False     
__E.image_out_path                               = './Images/' + __E.dataset_mode + '/' + __E.name
if not os.path.exists(__E.image_out_path):
    os.makedirs(__E.image_out_path)
else:
    shutil.rmtree(__E.image_out_path)
    os.makedirs(__E.image_out_path)

__E.num_test                                     = 500         # Number of images to test
__E.how_many_channel                             = 1           # Number of channel realizations per image
__E.epoch                                        = 'latest'    # Each model to use for testing
__E.load_iter                                    = 0


############################# OFDM configs ####################################

size_latent = (__E.size_w // (2**__E.n_downsample)) * (__E.size_h // (2**__E.n_downsample)) * (__E.C_channel // 2)
__E.P                                            = 1                                   # Number of symbols
__E.M                                            = 64                                  # Number of subcarriers per symbol
__E.K                                            = 16                                  # Length of CP
__E.L                                            = 8                                   # Number of paths
__E.decay                                        = 4                                   # Exponential decay for the multipath channel
__E.S                                            = size_latent // __E.M                # Number of packets

