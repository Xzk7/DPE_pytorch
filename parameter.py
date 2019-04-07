from multiprocessing import Process, Queue, Manager
from convnet import *
from function import *
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out

def str2bool(v):
    return v.lower() in ('true')

# Configure
def get_parameters():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='WGAN-v24-cycleganD2', choices=['WGAN-v24-cycleganD2'])
    parser.add_argument('--model_use_debug', type=str2bool, default=False)
    parser.add_argument('--num_exp', type=int, default=736)
    parser.add_argument('--num_gpu', type=int, default=4)
    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument('--version', type=str, default='WGAN-v24-cycleganD2_v1')

    # Training setting
    parser.add_argument('--netD_init_method', type=str, default='var_scale')
    parser.add_argument('--netD_init_times', type=int, default=0)
    parser.add_argument('--netD_init_weight', type=float, default=1e-3)
    parser.add_argument('--netD_lr', type=float, default=1e-5)
    parser.add_argument('--netD_lr_decay', type=float, default=75)
    parser.add_argument('--netD_lr_decay_epoch', type=int, default=75)
    parser.add_argument('--netD_buffer_times', type=int, default=50)
    parser.add_argument('--netD_regularization_weight', type=float, default=0.)
    parser.add_argument('--netG_init_weight', type=float, default=1e-3)
    parser.add_argument('--netD_times', type=int, default=50)
    parser.add_argument('--netG_lr', type=float, default=1e-5)
    parser.add_argument('--netG_lr_decay', type=float, default=0.95)
    parser.add_argument('--netG_lr_decay_epoch', type=int, default=75)
    parser.add_argument('--netG_regularization_weight', type=float, default=0.)
    parser.add_argument('--netD_times_grow', type=int, default=1)
    parser.add_argument('--loss_source_data_term', type=str, default='l2', choices=['l1','l2','PR','GD'])
    parser.add_argument('--loss_source_data_term_weight', type=float, default=1e3)
    parser.add_argument('--loss_constant_term', type=str, default='l2', choices=['l1', 'l2', 'PR', 'GD'])
    parser.add_argument('--loss_constant_term_weight', type=float, default=1e4)
    parser.add_argument('--loss_photorealism_is_our', type=str2bool, default=True)
    parser.add_argument('--loss_wgan_lambda', type=float, default=10.)
    parser.add_argument('--loss_wgan_lambda_grow', type=float, default=2.0)
    parser.add_argument('--loss_wgan_lambda_ignore', type=int, default=1)
    parser.add_argument('--loss_wgan_use_g_to_one', type=str2bool, default=False)
    parser.add_argument('--loss_wgan_gp_times', type=int, default=1)
    parser.add_argument('--loss_wgan_gp_use_all', type=str2bool, default=False)
    parser.add_argument('--loss_wgan_gp_bound', type=float, default=5e-2)
    parser.add_argument('--loss_wgan_gp_mv_decay', type=float, default=0.99)
    parser.add_argument('--loss_data_term_use_local_weight', type=str2bool, default=False)
    parser.add_argument('--loss_constant_term_use_local_weight', type=str2bool, default=False)
    parser.add_argument('--data_csr_buffer_size', type=int, default=1500)
    parser.add_argument('--sys_use_all_gpu_memory', type=str2bool, default=True)
    parser.add_argument('--loss_pr', type=str2bool, default=False)
    parser.add_argument('--loss_heavy', type=str2bool, default=True)
    parser.add_argument('--data_augmentation_size', type=int, default=8)
    parser.add_argument('--data_use_random_pad', type=str2bool, default=False)
    parser.add_argument('--data_train_batch_size', type=int, default=3)
    parser.add_argument('--load_previous_exp', type=int, default=0)
    parser.add_argument('--load_previous_epoch', type=int, default=0)
    parser.add_argument('--process_run_first_testing_epoch', type=str2bool, default=True)
    parser.add_argument('--process_write_test_img_count', type=int, default=498)
    parser.add_argument('--process_train_log_interval_epoch', type=int, default=20)
    parser.add_argument('--process_test_log_interval_epoch', type=int, default=2)
    parser.add_argument('--process_max_epoch', type=int, default=150, help='epoch num')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=True)
    parser.add_argument('--gpus', type=str, default='0,1', help='gpuids eg: 0,1,2,3  --parallel True')
    parser.add_argument('--dataset', type=str, default='MIT_Adobe', choices=['MIT_Adobe', 'MIT_Adobe&HDR'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--img_root1', type=str, default='./MIT_datasets/A', choices=['./MIT_datasets/A','./HDR_datasets'], \
                        help='root directory that contains images')
    parser.add_argument('--img_root2', type=str, default='./MIT_datasets/C',
                        choices=['./MIT_datasets/C', './HDR_datasets'], \
                        help='root directory that contains images')
    parser.add_argument('--train_file1', type=str, default='train_input.txt',
                        help='txt file that contains training filename')
    parser.add_argument('--train_file2', type=str, default='train_label.txt',
                        help='txt file that contains training filename')
    parser.add_argument('--test_file', type=str, default='test_label.txt',
                        help='txt file that contains training filename')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')

    # Step size
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=5)

    return parser.parse_args()

def flatten_list(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten_list(x))
    else:
        result.append(xs)
    return result

class NetInfo(object):
    def __init__(self, name):
        self.CONV_NETS = []
        if name == "netG":
            nonlinearity = selu_layer()
            net_1 = dict(net_name = '%s_1' % name, trainable = True)
            net_1['input_index'] = 0
            net_1['layers'] = flatten_list([
                conv_layer(3, 16, 3, 1, 1), nonlinearity, bn_layer(16), \
                conv_layer(16, 32, 5, 2, 2), nonlinearity, bn_layer(32), \
                conv_layer(32, 64, 5, 2, 2), nonlinearity, bn_layer(64), \
                conv_layer(64, 128, 5, 2, 2), nonlinearity, bn_layer(128), \
                conv_layer(128, 128, 5, 2, 2), nonlinearity, bn_layer(128), \
            ])
            self.CONV_NETS.append(net_1)

            net_2 = dict(net_name = '%s_2' % name, trainable = True)
            net_2['input_index'] = 15
            net_2['layers'] = flatten_list([
                conv_layer(128, 128, 5, 2, 2), nonlinearity, bn_layer(128), \
                conv_layer(128, 128, 5, 2, 2), nonlinearity, bn_layer(128), \
                conv_layer(128, 128, 8, 1, 0), nonlinearity, \
                conv_layer(128, 128, 1, 1, 0)\
            ])
            self.CONV_NETS.append(net_2)

            net_3 = dict(net_name='%s_3' % name, trainable=True)
            net_3['input_index'] = 15
            net_3['layers'] = flatten_list([ \
                conv_layer(128, 128, 3, 1, 1), global_concat_layer(24), \
                conv_layer(256, 128, 1, 1, 0), nonlinearity, bn_layer(128), \
                conv_layer(128, 128, 3, 1, 1), resize_layer(2), concat_layer(10), nonlinearity, bn_layer(256),\
                conv_layer(256, 128, 3, 1, 1), resize_layer(2), concat_layer(7), nonlinearity, bn_layer(192),\
                conv_layer(192, 64, 3, 1, 1), resize_layer(2), concat_layer(4), nonlinearity, bn_layer(96), \
                conv_layer(96, 32, 3, 1, 1), resize_layer(2), concat_layer(1), nonlinearity, bn_layer(48),\
                conv_layer(48, 16, 3, 1, 1), nonlinearity, bn_layer(16),\
                conv_layer(16, 3, 3, 1, 1), res_layer(0) \
                # , clip_layer() \
            ])
            self.CONV_NETS.append(net_3)

        elif name == "netD":
            nonlinearity = lrelu_layer(0.2)
            net_1 = dict(net_name = '%s_1' % name, trainable = True)
            net_1['input_index'] = 0
            net_1['layers'] = flatten_list([ \
                conv_layer(3, 16, 3, 1, 1), nonlinearity, in_layer(16), \
                conv_layer(16, 32, 5, 2, 2), nonlinearity, in_layer(32), \
                conv_layer(32, 64, 5, 2, 2), nonlinearity, in_layer(64), \
                conv_layer(64, 128, 5, 2, 2), nonlinearity, in_layer(128), \
                conv_layer(128, 128, 5, 2, 2), nonlinearity, in_layer(128), \
                conv_layer(128, 128, 5, 2, 2), nonlinearity, in_layer(128), \
                conv_layer(128, 1, 16, 1, 0), reduce_mean_layer()
            ])
            self.CONV_NETS.append(net_1)

def init_weights_netD(m):
    config = get_parameters()
    init_w = config.netD_init_weight
    rw = config.netD_regularization_weight
    if config.netD_init_method == "var_scale":
        if isinstance(m, nn.Conv2d):
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
            torch.nn.init.normal_(m.weight.data, 0.0, np.sqrt(init_w/fan_in))
            m.bias.data.fill_(0)
        elif(isinstance(m, nn.InstanceNorm2d)):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)
    elif config.netD_init_method == "rand_uniform":
        if(hasattr(m, 'weight')):
            torch.nn.init.uniform_(m.weight.data, -init_w * np.sqrt(3), init_w * np.sqrt(3))
            torch.nn.init.uniform_(m.bias.data, 0.0, -init_w * np.sqrt(3), init_w * np.sqrt(3))
    elif config.netD_init_method == "rand_normal":
        if(hasattr(m, 'weight')):
            torch.nn.init.normal_(m.weight.data, mean=0., std=init_w)
            torch.nn.init.normal_(m.bias.data, mean=0., std=init_w)
    else:
        if(hasattr(m, 'weight')):
            torch.nn.init.normal_(m.weight.data, mean=0., std=init_w)
            torch.nn.init.normal_(m.bias.data, mean=0., std=init_w)

def init_weights_netG(m):
    config = get_parameters()
    init_w = config.netG_init_weight
    rw = config.netD_regularization_weight
    if config.netD_init_method == "var_scale":
        if isinstance(m, nn.Conv2d):
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
            torch.nn.init.normal_(m.weight.data, 0.0, np.sqrt(init_w/fan_in))
            m.bias.data.fill_(0)
        elif(isinstance(m, nn.BatchNorm2d)):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)
    elif config.netD_init_method == "rand_uniform":
        if(hasattr(m, 'weight')):
            torch.nn.init.uniform_(m.weight.data, -init_w * np.sqrt(3), init_w * np.sqrt(3))
            m.bias.data.fill_(0)
    elif config.netD_init_method == "rand_normal":
        if(hasattr(m, 'weight')):
            torch.nn.init.normal_(m.weight.data, mean=0., std=init_w)
            m.bias.data.fill_(0)
    else:
        if(hasattr(m, 'weight')):
            torch.nn.init.normal_(m.weight.data, mean=0., std=init_w)
            m.bias.data.fill_(0)


