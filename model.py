import torch
import torch.nn as nn
from convnet import *
from parameter import *
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d)) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)

def conv_block(net_n):
    net_name = net_n['name']
    if net_name == 'conv':
        net = exe_conv_layer(net_n)
    elif net_name == 'selu':
        net = exe_selu_layer()
    elif net_name == 'relu':
        net = nn.ReLU()
    elif net_name == 'lrelu':
        net = exe_lrelu_layer(net_n)
    elif net_name == 'bn':
        net = exe_bn_layer(net_n)
    elif net_name == 'in':
        net = exe_in_layer(net_n)
    elif net_name == 'g_concat':
        net = net_n
    elif net_name == 'concat':
        net = net_n
    elif net_name == 'resize':
        net = net_n
    elif net_name == 'res_layer':
        net = net_n
    elif net_name == 'in':
        net = exe_in_layer(net_n)
    elif net_name == 'fc':
        net = net_n
        net['net_work'] = exe_fc_layer(net_n)
    elif net_name == 'reduce_mean':
        net = net_n
    else:
        assert False, 'Error layer name = %s' % net_name
    return net

# U-net
class Generator(nn.Module):
    def __init__(self, net_info):
        super(Generator, self).__init__()
        self.net_info = net_info
        self.scale = 1.0026538655307724
        self._make_layer()
        self.apply(init_weights_netG)

    def _make_layer(self):
        self.layer = []
        self.module = []
        self.module = nn.Sequential()
        for net_dict in self.net_info.CONV_NETS:
            for net_n in net_dict['layers']:
                net = conv_block(net_n)
                self.layer.append(net)
                if isinstance(net, dict):
                    pass
                elif isinstance(net, list):
                    self.module.append(net[0])
                    self.module.append(net[1])
                else:
                    if (net_n['name'] == 'selu'):
                        self.module.append(nn.SELU())
                    else:
                        self.module.append(net)
        self.module = nn.Sequential(nn.ModuleList(self.module))

    def forward(self, input_dict):
        input = input_dict['img']
        retouched = input_dict['retouched']
        result_list = [input]
        for i, block in enumerate(self.layer):
            if isinstance(block, nn.ELU):
                input = self.scale * block(input)
            elif isinstance(block, dict):
                if(block['name'] == 'g_concat'):
                    global_feature = result_list[block['index']]
                    global_feature = global_feature.repeat(1, 1, 32, 32)
                    input = result_list[self.net_info.CONV_NETS[1]['input_index']]
                    input = torch.cat((input, global_feature), 1)
                elif(block['name'] == 'concat'):
                    local_feature = result_list[block['index']]
                    input = torch.cat((input, local_feature), 1)
                elif(block['name'] == 'resize'):
                    input = nn.Upsample(scale_factor=block['scale'])(input)
                else:
                    input = input + result_list[block['index']]
            elif isinstance(block, list):
                if(not retouched):
                    input = block[0](input)
                else:
                    input = block[1](input)
            else:
                input = block(input)
            result_list.append(input)

        return input, result_list

'''netG = NetInfo("netG")
G1 = Generator(netG)
G2 = Generator(netG)
G2.load_state_dict(G1.state_dict())
print(id(G1))
print(id(G2))'''
'''model_parameters = filter(lambda p: p.requires_grad, G.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('|  Number of G Parameters: ' + str(params))'''

class Discriminator(nn.Module):
    def __init__(self, net_info):
        super(Discriminator, self).__init__()
        self.net_info = net_info
        self._make_layer()
        self.apply(init_weights_netD)

    def _make_layer(self):
        self.layer = []
        self.module = []
        for net_dict in self.net_info.CONV_NETS:
            for net_n in net_dict['layers']:
                net = conv_block(net_n)
                self.layer.append(net)
                if isinstance(net, dict):
                    pass
                elif isinstance(net, list):
                    self.module.append(net[0])
                else:
                    if (net_n['name'] == 'selu'):
                        self.module.append(nn.SELU())
                    else:
                        self.module.append(net)
        self.module = nn.Sequential(nn.ModuleList(self.module))

    def forward(self, input):
        input_list = [input]
        for i, block in enumerate(self.layer):
            if isinstance(block, dict):
                if block['name'] == 'fc':
                    input = input.resize(input.size(0), -1)
                    input = block['net_work'](input)
                else:
                    input = input.reshape(input.size(0), -1)
                    input = torch.mean(input, 1)
                    input = input.unsqueeze(-1)
            else:
                input = block(input)
            input_list.append(input)
        return input, input_list


'''model_parameters = filter(lambda p: p.requires_grad, G.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('|  Number of G Parameters: ' + str(params))
model_parameters = filter(lambda p: p.requires_grad, D.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('|  Number of D Parameters: ' + str(params))

model_parameters = filter(lambda p: True, G.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('|  Number of G Parameters: ' + str(params))
model_parameters = filter(lambda p: True, D.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('|  Number of D Parameters: ' + str(params))'''
