from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C


############################# Basic settings ####################################

__C.name                                         = 'JSCC_OFDM'      # Name of the experiment
__C.gpu_ids                                      = [1]              # GPUs to use
__C.dataset_mode                                 = 'OpenImage'  # ['CIFAR10', 'CIFAR100', 'CelebA', 'OpenImage']
__C.checkpoints_dir                              = './Checkpoints/' + __C.dataset_mode   # Path to store the model
__C.model                                        = 'JSCCOFDM'
__C.C_channel                                    = 12         # Number of channels for output latents (controls the communication rate)
                                                              # Calculation of the rate (channel usage per pixel): 
                                                              #           C_channel / (3 x 2^(2 x n_downsample + 1))                                                            
__C.SNR                                          = 5          # Signal to noise ratio
__C.SNR_cal                                      = 'ins'      # ['ins', 'avg']. 'ins' is for instantaneous SNR, 'avg' is for average SNR
__C.feedforward                                  = 'OFDM-CE-sub-EQ-sub'  # Different schemes: 
                                                                         # OFDM-CE-EQ: MMSE channel estimation and equalization without any subnets
                                                                         # OFDM-CE-sub-EQ: MMSE channel estimation and equalization with CE subnet
                                                                         # OFDM-CE-sub-EQ-sub: MMSE channel estimation and equalization with CE & EQ subnet
                                                                         # OFDM-feedback: pre-coding scheme with CSI feedback
__C.N_pilot                                      = 1          # Number of pilot symbols
__C.is_clip                                      = False      # Whether to apply signal clipping or not
__C.CR                                           = 1.2        # Clipping ratio if clipping is applied
__C.lam_h                                        = 50         # Weight for the channel reconstruction loss
__C.gan_mode                                     = 'none'     # ['wgangp', 'lsgan', 'vanilla', 'none']
__C.lam_G                                        = 0.02       # Weight for the adversarial loss
__C.lam_L2                                       = 100        # Weight for image reconstruction loss

############################# Model and training configs ####################################

if __C.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __C.n_layers_D                               = 3          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 2          # Downsample times
    __C.n_blocks                                 = 2          # Numebr of residual blocks
    

elif __C.dataset_mode == 'CelebA':
    __C.n_layers_D                               = 3          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3          # Downsample times
    __C.n_blocks                                 = 2          # Numebr of residual blocks


elif __C.dataset_mode == 'OpenImage':
    __C.n_layers_D                               = 4          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3          # Downsample times
    __C.n_blocks                                 = 4          # Numebr of residual blocks
    

__C.norm_D                                       = 'instance' if __C.gan_mode == 'wgangp' else 'batch'   # Type of normalization in Discriminator
__C.norm_EG                                      = 'batch'        # Type of normalization in Encoder and Generator



############################# Display and saving configs ####################################

__C.name = f'C{__C.C_channel}_{__C.feedforward}_SNR_{__C.SNR}_{__C.SNR_cal}_pilot_{__C.N_pilot}_hloss_{__C.lam_h}'
__C.name += f'_clip_{__C.CR}' if __C.is_clip else ''
__C.name += f'_{__C.gan_mode}_{__C.lam_G}' if __C.gan_mode != 'none' else ''
        
       



