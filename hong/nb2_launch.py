def map_BN_scale_relu_conv(idxs,pkey,dconv=False):
    b,s,c = idxs
    BSC = [
        ('BatchNorm{}'.format(b),0,'{}.0.running_mean'.format(pkey)),
        ('BatchNorm{}'.format(b),1,'{}.0.running_var'.format(pkey)),
        ('Scale{}'.format(s),0,'{}.0.weight'.format(pkey)),
        ('Scale{}'.format(s),1,'{}.0.bias'.format(pkey))
    ]
    if not dconv:
        BSC.extend([
            ('Convolution{}'.format(c),0,'{}.2.weight'.format(pkey)),
            ('Convolution{}'.format(c),1,'{}.2.bias'.format(pkey))
        ])
    else:
        BSC.extend([
            ('Deconvolution{}'.format(c),0,'{}.2.weight'.format(pkey))
        ])
    return BSC


def map_conv_BN_scale_relu(idxs,pkey,dconv=False):
    b,s,c = idxs
    bsc = []
    if not dconv:
        bsc.extend([
            ('Convolution{}'.format(c),0,'{}.0.weight'.format(pkey)),
            ('Convolution{}'.format(c),1,'{}.0.bias'.format(pkey))
        ])
    else:
        bsc.append(
            ('Deconvolution{}'.format(c),0,'{}.0.weight'.format(pkey))
        )

    bsc.extend([
        ('BatchNorm{}'.format(b),0,'{}.1.running_mean'.format(pkey)),
        ('BatchNorm{}'.format(b),1,'{}.1.running_var'.format(pkey)),
        ('Scale{}'.format(s),0,'{}.1.weight'.format(pkey)),
        ('Scale{}'.format(s),1,'{}.1.bias'.format(pkey)),
    ])
    return bsc

# print(map_BN_scale_relu_conv((1,1,1),'data_f'))
# print(map_conv_BN_scale_relu((2,2,2),'stack1.0'))

import _init_paths
import tools
import os.path as osp
import numpy as np
# from caffe import layers as L, params as P, to_proto
import matplotlib.pyplot as plt
from PIL import Image
import global_var as GV
import matplotlib.image as mping
import h5py
import scipy.io as scio
import os

from hong import InputLayer,ResizeToSameSize,EuclideanLossLayer
import torch
from torch import nn
from pdb import set_trace

def BN_scale_relu_conv(split, bottom, nout, ks=3, stride=1, pad=1, dilation = 1, conv_type = 'conv', in_place = False, lr = 2):
    nin = bottom.nout
    layers = []
    if dilation != 1 and conv_type == 'conv':
        pad = ((dilation - 1) * (ks - 1) + ks - 1) / 2 
    if split == 'train':
        layers.append(nn.BatchNorm2d(num_features=nin, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
    else:
        layers.append(nn.BatchNorm2d(num_features=nin, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    layers.append(nn.ReLU(inplace=True))
    
    if conv_type == 'conv':
        layers.append(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=ks, stride=stride, padding=pad, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
        )
    elif conv_type == 'dconv':
        layers.append(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=ks, stride=stride, padding=pad, output_padding=0, groups=1, bias=False, dilation=dilation, padding_mode='zeros')
        )
    BN_scale, relu, conv = nn.Sequential(*layers[:-2]), nn.Sequential(*layers[:-1]), nn.Sequential(*layers)
    BN_scale.nout = nin
    relu.nout = nin
    conv.nout = nout

    return BN_scale, BN_scale, relu, conv

def conv_BN_scale_relu(split, bottom, nout, ks=3, stride=1, pad=1, dilation = 1, conv_type = 'conv', in_place = True, lr = 2):
    layers = []
    nin = bottom.nout
    if dilation != 1 and conv_type == 'conv':
        pad = ((dilation - 1) * (ks - 1) + ks - 1) / 2 
    if conv_type == 'conv':
        layers.append(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=ks, stride=stride, padding=pad, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
        )
    elif conv_type == 'dconv':
        layers.append(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=ks, stride=stride, padding=pad, output_padding=0, groups=1, bias=False, dilation=dilation, padding_mode='zeros')
        )
        
    if split == 'train':
        layers.append(
            nn.BatchNorm2d(num_features=nout, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )
    else:
        layers.append(
            nn.BatchNorm2d(num_features=nout, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    layers.append(nn.ReLU(inplace=in_place))
    Ms = (nn.Sequential(*layers[:-2]), nn.Sequential(*layers[:-1]), nn.Sequential(*layers))
    for m in Ms:
        m.nout = nout
    conv, BN_scale, relu = Ms
    return conv, BN_scale, BN_scale, relu

def max_pool(bottom, ks=2, stride=2):
    nin = bottom.nout
    M = nn.MaxPool2d(kernel_size=ks, stride=stride, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    M.nout = nin
    return M

test_num = 54
GV.test_data_dir = '0408data1/Images_train'  # HERE！
GV.test_labels_dir = '0408data1/GT_train'
GV.lr = 0.1

class SeqSkip(nn.Module):
    def __init__(self, m1,m2):
        super(SeqSkip, self).__init__()
        self.m1=m1
        self.m2=m2
        self.nout=m2.nout
    def forward(self,result):
        result1 = self.m1(result)
        result2 = self.m2(result1)
        result = result2 + result
        return result

class SeqMSkip(nn.Module):
    def __init__(self, m1,m2,mm):
        super(SeqMSkip, self).__init__()
        self.m1=m1
        self.m2=m2
        self.mm=mm
        self.nout=m2.nout
    def forward(self,x):
        x1=self.m1(x)
        x2=self.m2(x1)
        xm=self.mm(x)
        return xm+x2

class SimpleNN(nn.Module):
    def __init__(self, device, dtype=torch.float32, split='test'):
        super(SimpleNN, self).__init__()

        # B,S,C = 0,0,0
        self.TotalMaps = []
        TotalMaps=[]
        
        nout = 32
        repeat = 2
        dilation = 1
        ks = 3
        pad = 1
        GV.repeat = repeat  # HOLD
        self.nout = nout
        self.device=device
      
        self.input_layer = InputLayer(
               params = dict(split=split, 
                                    data_dir = '/mnt/pub_workspace_2T/hong_data/detect/MEnet/drive/MEnet',
                                    train_data_dir = '0408data1/Images_train',
                                    train_sobel_dir = '0408data1/Images_train',
                                    train_labels_dir = '0408data1/GT_train',
                                    test_data_dir = GV.test_data_dir,#
                                    test_sobel_dir = '0408data1/Images_train',
                                    test_labels_dir = GV.test_labels_dir,#
                                    batch_size = 1,
                                    test_batch_size = 1,
                                    resize_size = [224, 224])
        )
        set_dilation = 1       
        class A():    pass
        data=A()
        data.nout = 3
        result = data

        # BRANCH from data
        _, _, _, self.data_f = BN_scale_relu_conv(split, bottom = data, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        TotalMaps.extend(map_BN_scale_relu_conv((1,1,1),'data_f'))
        # BRANCH
        result_s = []

        result_s.append(
            conv_BN_scale_relu(split, bottom = result, nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)[-1]
        )
        TotalMaps.extend(map_conv_BN_scale_relu((2,2,2),'stack1.0'))

        offset=1
        name='stack1'
        idxs = [ (6,6,6),(5,5,5),(4,4,4),(3,3,3) ]
        for i in range(repeat): #224
            _, _, _, M1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
            realname = '{0}.{1}.m1'.format(name,i+offset)
            TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

            _, _, _, M2 = BN_scale_relu_conv(split, bottom = M1, nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
            realname = '{0}.{1}.m2'.format(name,i+offset)
            TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
            result_s.append(
                SeqSkip(M1,M2)
            )
        self.stack1 = nn.Sequential(*result_s) # start from data
        self.stack1.nout = self.stack1[-1].nout

        # BRANCH
        scale0 = self.stack1[-1]
        _, _, _, self.scale0_m = BN_scale_relu_conv(split, bottom = scale0, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        TotalMaps.extend(map_BN_scale_relu_conv((7,7,7),'scale0_m'))

        # BRANCH
        result_s = []
        idxs=(8,9,10,11,12)
        idxs = [ (i,)*3 for i in idxs[::-1]]
        name='scale1'
        for i in range(repeat):#112
            if i == 0:
                _, _, _, result1 = BN_scale_relu_conv(split, bottom = self.stack1, nout = 2 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)  
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                
                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.stack1, nout = 2 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                
                result_s.append(
                    SeqMSkip(result1, result2, resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqSkip(result1,result2)
                )
        
        # BRANCH
        self.scale1 = nn.Sequential(*result_s)  # from stack1
        self.scale1.nout = self.scale1[-1].nout

        a, b, c, self.scale1_m = BN_scale_relu_conv(split, bottom = self.scale1, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        idx=(13,13,13)
        realname='scale1_m'
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))

        # BRANCH
        result_s = []
        idxs = (14,15,16,17,18)
        idxs = list((i,)*3 for i in idxs[::-1])
        # set_trace()
        name = 'scale2'
        for i in range(repeat):#56
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale1, nout = 4 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.scale1, nout = 4 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                result_s.append(
                    SeqSkip(result1, result2)
                )

        self.scale2 = nn.Sequential(*result_s)  # from scale1
        self.scale2.nout = self.scale2[-1].nout

#         scale2 = result  # BRANCH
        idx=(19,19,19)
        realname='scale2_m'
        a, b, c, self.scale2_m = BN_scale_relu_conv(split, bottom = self.scale2, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        
        # BRANCH
        result_s=[]
        idxs = (20,21,22,23,24)
        idxs = list((i,)*3 for i in idxs[::-1])
        name = 'scale3'
        for i in range(repeat):#28
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale2, nout = 8 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.scale2, nout = 8 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqMSkip(result1, result2, resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                result_s.append(
                    SeqSkip(result1,result2)
                )
        # BRANCH
        self.scale3 = nn.Sequential(*result_s)  # from scale2
        self.scale3.nout = self.scale3[-1].nout
        
        # BRANCH
        a, b, c, self.scale3_m = BN_scale_relu_conv(split, bottom = self.scale3, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        idx=(25,25,25)
        realname='scale3_m'
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        
        result_s = []
        idxs = list(range(26,31))
        idxs = list((i,)*3 for i in idxs[::-1])
        name = 'scale4'
        for i in range(repeat):#14
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale3, nout = 16 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.scale3, nout = 16 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                result_s.append(
                    SeqSkip(result1,result2)
                )
        # BRANCH
        self.scale4 = nn.Sequential(*result_s)  # from scale3
        self.scale4.nout = self.scale4[-1].nout

        a, b, c, self.scale4_m = BN_scale_relu_conv(split, bottom = self.scale4, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        idx=(31,31,31)
        realname='scale4_m'
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        
        result_s=[]
        idxs = list(range(32,37))
        idxs = list((i,)*3 for i in idxs[::-1])
        name = 'scale5'
        for i in range(repeat):#7
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale4, nout = 32 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 32 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.scale4, nout = 32 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 32 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 32 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqSkip(result1,result2)
                )
        # BRANCH
        self.scale5 = nn.Sequential(*result_s)
        self.scale5.nout = self.scale5[-1].nout
        
        # BRANCH
        a, b, c, self.scale5_m = BN_scale_relu_conv(split, bottom = self.scale5, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        idx = (37,37,37)
        realname='scale5_m'
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        dilation = 1

        # BRANCH
        a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale5, nout = 32 * nout, ks=7, stride=1, pad = 0, dilation = dilation, in_place = False)
        idx = (38,38,38)
        realname='scale5_5.0'
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))

        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 32 * nout, ks =7, stride=1, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
        realname='scale5_5.1'
        idx=(39,39,1)
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname,dconv=True))
        


        self.scale5_5 = nn.Sequential(result1,result2)  # fron scale5
        self.scale5_5.nout = self.scale5_5[-1].nout

        a, b, c, self.scale5_u = BN_scale_relu_conv(split, bottom = self.scale5_5, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        realname='scale5_u'
        idx = (70,70,59)
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))

        # BRANCH TEMPORAL
        self.scale5_5.nout += self.scale5.nout
        
        result_s = []
        idxs = [
            (40,40,2),
            (41,41,39),
            (42,42,3),
            (43,43,40),
            (44,44,41)
        ]
        idxs = idxs[::-1]
        name = 'stack6'
        for i in range(repeat):#14
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale5_5, nout = 16 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation ,in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.scale5_5, nout = 16 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))
                
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation,in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation ,in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqSkip(result1, result2)
                )

        self.stack6=nn.Sequential(*result_s)  # from Concat(self.scale5_5, self.scale5)
        self.stack6.nout = self.stack6[-1].nout
        
        # BRANCH
        a, b, c, self.scale4_u = BN_scale_relu_conv(split, bottom = self.stack6, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        realname='scale4_u'
        idx = (69,69,58)
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        
        # BRANCH TEMPORAL
        self.stack6.nout += self.scale4.nout
        
        # BRANCH
        result_s=[]
        idxs = [
            (45,45,4),
            (47,47,5),
            (46,46,42),
            (48,48,43),
            (49,49,44)
        ]
        idxs = idxs[::-1]
        name = 'stack7'
        for i in range(repeat):#28
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.stack6, nout = 8 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.stack6, nout = 8 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation,in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqSkip(result1, result2)
                )
        
        self.stack7=nn.Sequential(*result_s)  # BRANCH
        self.stack7.nout = self.stack7[-1].nout
        
        # BRANCH
        a, b, c, self.scale3_u = BN_scale_relu_conv(split, bottom = self.stack7, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        realname='scale3_u'
        idx = (68,68,57)
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        
        # BRANCH TEMPORAL
        self.stack7.nout += self.scale3.nout
        
        result_s=[]
        idxs = [
            (50,50,6),
            (52,52,7),
            (51,51,45),
            (53,53,46),
            (54,54,47)
        ]
        idxs = idxs[::-1]
        name = 'stack8'
        for i in range(repeat):#56
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.stack7, nout = 4 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.stack7, nout = 4 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqSkip(result1,result2)
                )
        self.stack8=nn.Sequential(*result_s)
        self.stack8.nout = self.stack8[-1].nout
        
        a, b, c, self.scale2_u = BN_scale_relu_conv(split, bottom = self.stack8, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        realname='scale2_u'
        idx = (67,67,56)
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        
        # BRANCH TEMPORAL
        self.stack8.nout += self.scale2.nout
        
        result_s=[]
        idxs = [
            (55,55,8),
            (57,57,9),
            (56,56,48),
            (58,58,49),
            (59,59,50)
        ]
        idxs = idxs[::-1]
        name = 'stack9'
        for i in range(repeat):#112
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.stack8, nout = 2 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.stack8, nout = 2 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))
                
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqSkip(result1,result2)
                )
        
        self.stack9=nn.Sequential(*result_s)
        self.stack9.nout=self.stack9[-1].nout

        a, b, c, self.scale1_u = BN_scale_relu_conv(split, bottom = self.stack9, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        realname='scale1_u'
        idx = (66,66,55)
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        
        # BRANCH TEMPORAL
        self.stack9.nout += self.scale1.nout
        
        result_s = []
        idxs = [
            (60,60,10),
            (62,62,11),
            (61,61,51),
            (63,63,52),
            (64,64,53)
        ]
        idxs = idxs[::-1]
        name = 'stack10'      
        for i in range(repeat):#224
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.stack9, nout = 1 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.stack9, nout = 1 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                realname = '{0}.{1}.mm'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname,dconv=True))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 1 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqMSkip(result1, result2, resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 1 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                realname = '{0}.{1}.m1'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 1 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                realname = '{0}.{1}.m2'.format(name,i)
                TotalMaps.extend(map_BN_scale_relu_conv(idxs.pop(),realname))

                result_s.append(
                    SeqSkip(result1,result2)
                )
    
        self.stack10=nn.Sequential(*result_s)
        self.stack10.nout=self.stack10[-1].nout
        
        a, b, c, self.scale0_u = BN_scale_relu_conv(split, bottom = self.stack10, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        realname='scale0_u'
        idx = (65,65,54)
        TotalMaps.extend(map_BN_scale_relu_conv(idx,realname))
        
        self.resize = ResizeToSameSize(device, dtype)
        self.resize.nout = sum(layer.nout for layer in (self.data_f, self.scale0_m, self.scale1_m, self.scale2_m, self.scale3_m, self.scale4_m, self.scale5_m, self.scale0_u, self.scale1_u, self.scale2_u, self.scale3_u, self.scale4_u, self.scale5_u))
        
        a, b, metric, c = conv_BN_scale_relu(split, bottom = self.resize, nout = 16, ks=3, stride=1, pad = 1, dilation = dilation, in_place = True)
        self.metric = nn.Sequential(metric, nn.Tanh())
        realname='metric.0'
        idx = (71,71,60)
        TotalMaps.extend(map_conv_BN_scale_relu(idx,realname))
    
        a, b, self.sal, d = conv_BN_scale_relu(split, bottom = self.resize, nout = 2, ks=3, stride=1, pad = 1, dilation = dilation, in_place = True)
        realname='sal'
        idx = (72,72,61)
        TotalMaps.extend(map_conv_BN_scale_relu(idx,realname))
        #　ＨＥＲＥ!
        self.TotalMaps = TotalMaps

    def forward(self, data):
        # set_trace()
        data_f = self.data_f(data)
        stack1 = self.stack1(data)
        scale0_m = self.scale0_m(stack1)
        scale1 = self.scale1(stack1)
        scale1_m = self.scale1_m(scale1)
        scale2 = self.scale2(scale1)
        scale2_m = self.scale2_m(scale2)
        scale3 = self.scale3(scale2)
        scale3_m = self.scale3_m(scale3)
        scale4 = self.scale4(scale3)
        scale4_m = self.scale4_m(scale4)
        scale5 = self.scale5(scale4)
        scale5_m = self.scale5_m(scale5)
        scale5_5 = self.scale5_5(scale5)
        scale5_u = self.scale5_u(scale5_5)

        scale5_5 = torch.cat((scale5_5, scale5), 1)  # guessed C

        stack6 = self.stack6(scale5_5)
        scale4_u = self.scale4_u(stack6)

        stack6 = torch.cat((stack6,scale4),1)

        stack7 = self.stack7(stack6)
        scale3_u = self.scale3_u(stack7)
        
        stack7 = torch.cat((stack7, scale3),1)
        
        stack8 = self.stack8(stack7)
        scale2_u = self.scale2_u(stack8)

        stack8 = torch.cat((stack8,scale2),1)
        
        stack9 = self.stack9(stack8)
        scale1_u = self.scale1_u(stack9)

        stack9 = torch.cat((stack9,scale1),1)

        stack10 = self.stack10(stack9)
        scale0_u = self.scale0_u(stack10)
        
        resize = self.resize(
            data_f, scale0_m, scale1_m, scale2_m, scale3_m, scale4_m,
            scale5_m, scale0_u, scale1_u, scale2_u, scale3_u,
            scale4_u, scale5_u
        )

        metric = self.metric(resize)
        sal = self.sal(resize)

        return metric, sal

    def forward_without_input(self):
        data, label=self.get_input()
        return self.forward(data), label
    
    def get_input(self):
        return (torch.tensor(d, dtype=torch.float32) for d in self.input_layer())

    def clear(self):
        self.resize.clear()
