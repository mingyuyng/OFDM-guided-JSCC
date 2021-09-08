from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C


# Basic settings
__C.name                                         = 'JSCC_OFDM'      # Name of the experiment
__C.gpu_ids                                      = [0]              # GPUs to use
__C.dataset_mode                                 = 'CIFAR10'  # ['CIFAR10', 'CIFAR100', 'CelebA', 'Openimage']
__C.checkpoints_dir                              = './Checkpoints/' + __C.dataset_mode + '_OFDM'   # Path to store the model
__C.model                                        = 'JSCCOFDM'


# Model config for different datasets
if __C.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __C.n_layers_D                               = 3          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 2          # Downsample times
    __C.n_blocks                                 = 2          # Numebr of residual blocks
    __C.batch_size                               = 128
    __C.n_epochs                                 = 150        # Number of epochs without lr decay
    __C.n_epochs_decay                           = 150        # Number of epochs with lr decay
    __C.lr_policy                                = 'linear'   # decay policy.  
    __C.beta1                                    = 0.5        # parameter for ADAM
    __C.lr                                       = 5e-4       # Initial learning rate

    __C.dataroot                                 = './data'
    __C.sizew                                    = 32
    __C.sizeh                                    = 32


elif __C.dataset_mode == 'CelebA':
    __C.n_layers_D                               = 3          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3          # Downsample times
    __C.n_blocks                                 = 2          # Numebr of residual blocks
    __C.batch_size                               = 64
    __C.n_epochs                                 = 15         # Number of epochs without lr decay
    __C.n_epochs_decay                           = 15         # Number of epochs with lr decay
    __C.lr_policy                                = 'linear'   # decay policy.  Availability:  see options/train_options.py
    __C.beta1                                    = 0.5        # parameter for ADAM
    __C.lr                                       = 5e-4       # Initial learning rate
    
    __C.dataroot                                 = './data/celeba/CelebA_train'
    __C.load_size                                = 80
    __C.crop_size                                = 64
    __C.sizew                                    = 64
    __C.sizeh                                    = 64


elif __C.dataset_mode == 'OpenImage':
    __C.n_layers_D                               = 4          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3          # Downsample times
    __C.n_blocks                                 = 4          # Numebr of residual blocks
    __C.batch_size                               = 24
    __C.n_epochs                                 = 10         # Number of epochs without lr decay
    __C.n_epochs_decay                           = 10         # Number of epochs with lr decay
    __C.lr_policy                                = 'linear'   # decay policy.  Availability:  see options/train_options.py
    __C.beta1                                    = 0.5        # parameter for ADAM
    __C.lr                                       = 1e-4       # Initial learning rate
    
    __C.dataroot                                 = './data/opv6'
    __C.sizew                                    = 256
    __C.sizeh                                    = 256


__C.verbose = True
__C.serial_batches = False
__C.max_dataset_size = float("inf")
__C.num_threads = 4

__C.input_nc                                     = 3          # Number of channels for the input image
__C.output_nc                                    = 3          # Number of channels for the output image
__C.ngf                                          = 64         # Number of filters in the first conv layer
__C.max_ngf                                      = 256        # Maximum number of filters in any conv layer


############################################################
# Basic setups
############################################################
__C.C_channel                                    = 12         # Number of output latents before being reshaped
__C.SNR                                          = 5
__C.SNR_cal                                      = 'ins'      # ['ins', 'avg']. 'ins' is for instantaneous SNR, 'avg' is for average SNR
__C.feedforward                                  = 'OFDM-CE-sub-EQ-sub'  # [OFDM-CE-EQ, OFDM-CE-sub-EQ, OFDM-CE-sub-EQ-sub, OFDM-feedback]
__C.N_pilot                                      = 1          # Number of pilot symbols

__C.is_clip                                      = False      # Whether to apply signal clipping
__C.CR                                           = 1.2        # Clipping ratio

__C.lam_h                                        = 50         # Weight for the channel reconstruction loss
__C.gan_mode                                     = 'none'     # ['wgangp', 'lsgan', 'vanilla', 'none']
__C.lam_G                                        = 0.02       # Weight for the adversarial loss
__C.lam_L2                                       = 100        # Weight for image reconstruction loss
#############################################################


if __C.gan_mode == 'wgangp':
    __C.norm_D                                   = 'instance'   # Use instance normalization when using WGAN.  Available: 'instance', 'batch', 'none'
else:
    __C.norm_D                                   = 'batch'      # Used batch normalization otherwise

__C.norm_EG = 'batch'


# OFDM settings
size_latent = (__C.sizew // (2**__C.n_downsample)) * (__C.sizeh // (2**__C.n_downsample)) * (__C.C_channel // 2)
__C.P                                            = 1                                   # Number of symbols
__C.M                                            = 64                                  # Number of subcarriers per symbol
__C.K                                            = 16                                  # Length of CP
__C.L                                            = 8                                   # Number of paths
__C.decay                                        = 4
__C.S                                            = size_latent // __C.M                # Number of packets


# Display and training
__C.name = f'C{__C.C_channel}_{__C.feedforward}_SNR_{__C.SNR}_{__C.SNR_cal}_pilot_{__C.N_pilot}_hloss_{__C.lam_h}'

if __C.is_clip:
    __C.name += f'_clip_{__C.CR}'
if __C.gan_mode != 'none':
    __C.name += f'_{__C.gan_mode}_{__C.lam_G}'


__C.display_env                                  = __C.dataset_mode + '_OFDM_' + __C.name
__C.display_freq                                 = 50               # frequency of showing training results on screen
__C.display_ncols                                = 4                # if positive, display all images in a single visdom web panel with certain number of images per row
__C.display_id                                   = 1                # window id of the web display
__C.display_server                               = "http://localhost"  #visdom server of the web display     
__C.display_port                                 = 8998             # visdom port of the web display
__C.display_winsize                              = 256
__C.update_html_freq                             = 100              # frequency of saving training results to html
__C.print_freq                                   = 100              # frequency of showing training results on console
        
__C.save_latest_freq                             = 5000             #frequency of saving the latest results
__C.save_epoch_freq                              = 40               #frequency of saving checkpoints at the end of epochs
__C.save_by_iter                                 = True             #whether saves model by iteration
__C.continue_train                               = False            #continue training: load the latest model
__C.epoch_count                                  = 1                #the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...

        
       



