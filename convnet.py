import torch
import torch.nn as nn
import torch.nn.functional as F

PARAMETERS_NAME = ["conv_%d_w", \
                   "conv_%d_b", \
                   "prelu_%d_%d_alpha", \
                   "bn_%d_%d_offset", \
                   "bn_%d_%d_scale", \
                   "bn_%d_%d_mv_mean", \
                   "bn_%d_%d_mv_var", \
                   "in_%d_%d_offset", \
                   "in_%d_%d_scale", \
                   "ln_%d_%d_offset", \
                   "ln_%d_%d_scale"]

# relu
def relu_layer():
    return dict(name='relu')

def exe_relu_layer():
    return torch.nn.ReLU()

# prelu
def prelu_layer():
    return dict(name='prelu')

# lrelu
def lrelu_layer(leak):
    return dict(name='lrelu', leak=leak)

def exe_lrelu_layer(layer_o):
    leak = layer_o['leak']
    return nn.LeakyReLU(leak)

# selu
def selu_layer():
    return dict(name='selu')

def exe_selu_layer():
    # alpha = 1.6732632423543772848170429916717
    # scale = 1.0507009873554804934193349852946
    alpha, scale = (1.0198755295894968, 1.0026538655307724)
    return nn.ELU(alpha)

# bn
def bn_layer(out_channel, epsilon=1e-5,):
    return dict(
        name='bn',
        out = out_channel,
        epsilon=epsilon)

def exe_bn_layer(layer_o):
    return [nn.BatchNorm2d(num_features = layer_o['out'], eps = layer_o['epsilon']), nn.BatchNorm2d(num_features = layer_o['out'], eps = layer_o['epsilon'])]

# in
def in_layer(out_channel, epsilon=1e-5):
    return dict(
        name='in',
        out = out_channel,
        epsilon = epsilon
    )

def exe_in_layer(layer_o):
    return nn.InstanceNorm2d(num_features = layer_o['out'], eps = layer_o['epsilon'], affine=True)

# fc
def fc_layer(in_channel, out_channel = 1, bias = True):
    return dict(
        name = 'fc',
        in_channel = in_channel,
        out_channel = out_channel,
        bias = bias
    )

def exe_fc_layer(layer_o):
    return nn.Linear(in_features = layer_o['in_channel'], out_features = layer_o['out_channel'], bias = layer_o['bias'])

# reduce_mean
def reduce_mean_layer():
    return dict(name = 'reduce_mean')

# conv
def conv_layer(in_channel, out_channel, kernel_size, stride, padding):
    return dict(
        name = 'conv', in_channel = in_channel, out_channel = out_channel, kernel = kernel_size , stride = stride, padding = padding
    )

def exe_conv_layer(layer_o):
    return nn.Conv2d(in_channels = layer_o['in_channel'], out_channels = layer_o['out_channel'], kernel_size = layer_o['kernel'], stride = layer_o['stride'], padding = layer_o['padding'])

# concat
def concat_layer(index):
    return dict(name = 'concat', index = index)

# global concat
def global_concat_layer(index):
    return dict(name = 'g_concat', index = index)

# resize
def resize_layer(scale):
    return dict(
        name='resize',
        scale=scale)

# res
def res_layer(index):
    return dict(name = 'res_layer', index = index)




