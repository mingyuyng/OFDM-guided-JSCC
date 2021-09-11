from easydict import EasyDict as edict
from config import cfg

__T                                              = cfg

############################# Training configs ####################################

if __T.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __T.batch_size                               = 128        # Batch size
    __T.serial_batches                           = False      # The batches are continuous or randomly shuffled
    __T.n_epochs                                 = 150        # Number of epochs without lr decay
    __T.n_epochs_decay                           = 150        # Number of epochs with lr decay
    __T.lr_policy                                = 'linear'   # decay policy.  
    __T.beta1                                    = 0.5        # parameter for ADAM
    __T.lr                                       = 5e-4       # Initial learning rate

elif __T.dataset_mode == 'CelebA':
    __T.batch_size                               = 64         # Batch size
    __T.serial_batches                           = False      # The batches are continuous or randomly shuffled
    __T.n_epochs                                 = 15         # Number of epochs without lr decay
    __T.n_epochs_decay                           = 15         # Number of epochs with lr decay
    __T.lr_policy                                = 'linear'   # decay policy.  Availability:  see options/train_options.py
    __T.beta1                                    = 0.5        # parameter for ADAM
    __T.lr                                       = 5e-4       # Initial learning rate

elif __T.dataset_mode == 'OpenImage':
    __T.batch_size                               = 24         # Batch size
    __T.serial_batches                           = False      # The batches are continuous or randomly shuffled
    __T.n_epochs                                 = 10         # Number of epochs without lr decay
    __T.n_epochs_decay                           = 10         # Number of epochs with lr decay
    __T.lr_policy                                = 'linear'   # decay policy.  Availability:  see options/train_options.py
    __T.beta1                                    = 0.5        # parameter for ADAM
    __T.lr                                       = 1e-4       # Initial learning rate
    

__T.print_freq                                   = 100              # frequency of showing training results on console   
__T.save_latest_freq                             = 5000             #frequency of saving the latest results
__T.save_epoch_freq                              = 40               #frequency of saving checkpoints at the end of epochs
__T.save_by_iter                                 = True             #whether saves model by iteration
__T.continue_train                               = False            #continue training: load the latest model
__T.epoch_count                                  = 1                #the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
__T.verbose                                      = False
__T.isTrain                                      = True
        
       



