from easydict import EasyDict as edict
from config import cfg

__E                                              = cfg

# Model config for different datasets
if __E.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __E.batch_size                               = 1

elif __E.dataset_mode == 'CelebA':
    __E.batch_size                               = 1

elif __E.dataset_mode == 'OpenImage':
    __E.batch_size                               = 24
    __E.dataroot                                 = './data/opv6'

__E.verbose = False
__E.serial_batches = True
__E.isTrain                                      = False     
__E.image_out_path                               = './Images/' + opt.dataset_mode + '_OFDM/' + opt.name
if os.path.exists(__E.image_out_path) == False:
    os.makedirs(__E.image_out_path)
else:
    shutil.rmtree(__E.image_out_path)
    os.makedirs(__E.image_out_path)

__E.num_test                                     = 500         # Number of images to test
__E.how_many_channel                             = 5           # Number of channel realizations per image
