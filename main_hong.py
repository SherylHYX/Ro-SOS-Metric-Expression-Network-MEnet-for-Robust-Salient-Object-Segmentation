import _init_paths
# import caffe
import tools
import os.path as osp
import numpy as np
from caffe import layers as L, params as P, to_proto
import matplotlib.pyplot as plt
from PIL import Image
import global_var as GV
import matplotlib.image as mping
import h5py
import scipy.io as scio
import os

from MEnet.hong import InputLayer,ResizeToSameSize,EuclideanLossLayer
from torch import nn
#das

def BN_scale_relu_conv(split, bottom, nout, ks=3, stride=1, pad=1, dilation = 1, conv_type = 'conv', in_place = False, lr = 2):
    nin = bottom.nout
    layers = []
    if dilation != 1 and conv_type == 'conv':
        pad = ((dilation - 1) * (ks - 1) + ks - 1) / 2 
    if split == 'train':
        layers.append(nn.BatchNorm2d(num_features=nin, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
    else:
        layers.append(nn.BatchNorm2d(num_features=nin, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        # BN = L.BatchNorm(bottom, batch_norm_param = dict(use_global_stats = True), in_place=in_place, 
        #                 param = [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
    # scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)
    layers.append(nn.ReLU(inplace=True))
    # relu = L.ReLU(scale, in_place=True) 
    
    if conv_type == 'conv':
        layers.append(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=ks, stride=stride, padding=pad, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
        )
        # conv = L.Convolution(relu, kernel_size=ks, stride=stride, dilation = dilation, num_output=nout, pad=pad, bias_term = True, #, std = 0.000000001, mean = 0
        #                     weight_filler = dict(type='xavier'),
        #                     bias_filler = dict(type='constant'),
        #                     param=[dict(lr_mult=lr/2, decay_mult=1), dict( lr_mult=lr, decay_mult=0)])
    elif conv_type == 'dconv':
        layers.append(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=ks, stride=stride, padding=pad, output_padding=0, groups=1, bias=False, dilation=dilation, padding_mode='zeros')
        )
        # conv = L.Deconvolution(relu, convolution_param=dict( weight_filler = dict(type='xavier'), dilation = dilation, num_output = nout, kernel_size = ks, stride = stride, pad = pad, bias_term=False), param=[dict(lr_mult=1)])
    M = nn.Sequential(*layers)
    M.nout=nout
    # BN, scale, relu, conv
    return M[0], M[0], M[1], M

def conv_BN_scale_relu(split, bottom, nout, ks=3, stride=1, pad=1, dilation = 1, conv_type = 'conv', in_place = True, lr = 2):
    layers = []
    nin = bottom.nout
    if dilation != 1 and conv_type == 'conv':
        pad = ((dilation - 1) * (ks - 1) + ks - 1) / 2 
#        ks = (dilation - 1) * (ks - 1) + ks
#        dilation = 1
    if conv_type == 'conv':
        layers.append(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=ks, stride=stride, padding=pad, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
        )
        # conv = L.Convolution(bottom, kernel_size=ks, stride=stride, dilation = dilation, num_output=nout, pad=pad, bias_term = True, #, std = 0.000000001, mean = 0
        #                     weight_filler = dict(type='xavier'),
        #                     bias_filler = dict(type='constant'),
        #                     param=[dict(lr_mult=lr/2, decay_mult=1), dict(lr_mult=lr, decay_mult=0)])
    elif conv_type == 'dconv':
        layers.append(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=ks, stride=stride, padding=pad, output_padding=0, groups=1, bias=False, dilation=dilation, padding_mode='zeros')
        )
        # conv = L.Deconvolution(bottom, convolution_param=dict( weight_filler = dict(type='xavier'), dilation = dilation, num_output = nout, kernel_size = ks, stride = stride, pad = pad, bias_term=False), param=[dict(lr_mult=1)])
        
    if split == 'train':
        layers.append(
            nn.BatchNorm2d(num_features=nin, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )
        # BN = L.BatchNorm(conv, batch_norm_param = dict(use_global_stats = False),  in_place=in_place,
        #                 param = [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
    else:
        layers.append(
            nn.BatchNorm2d(num_features=nin, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # BN = L.BatchNorm(conv, batch_norm_param = dict(use_global_stats = True), in_place=in_place, 
        #                 param = [dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
    # scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = in_place)
    layers.append(nn.ReLU(inplace=in_place))
    # relu = L.ReLU(scale, in_place=in_place)
    # return conv, BN, scale, relu
    M = nn.Sequential(*layers)
    M.nout = nout
    # conv, BN, scale, relu
    return M[0], M[1], M[1], M

def max_pool(bottom, ks=2, stride=2):
    nin = bottom.nout
    # return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
    M = nn.MaxPool2d(kernel_size=ks, stride=stride, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    M.nout = nin
    return M

test_num = 54
GV.test_data_dir = '0408data1/Images_train'  #'0322data/Train-n/Images_train' 0305_data_split  0331data/Train-n/Images_train

GV.test_labels_dir = '0408data1/GT_train'


GV.lr = 0.1

class SeqSkip(nn.Module):
    def __init__(self, m1,m2):
        super(SeqSkip, self).__init__()
        self.m1=m1
        self.m2=m2
        self.nout=m2.nout
    def forward(result):
        result1 = self.m1(result)
        result2 = self.m2(result1)
        result = result2 + result  # nout is the same with m2.nout
        return result

    
class SeqMSkip(nn.Module):
    # sequential module with modified skip connection
    def __init__(self, m1,m2,mm):
        super(SeqMSkip, self).__init__()
        self.m1=m1
        self.m2=m2
        self.mm=mm
        self.nout=m2.nout
    def forward(x):
        x1=self.m1(x)
        x2=self.m2(x1)
        xm=self.mm(x)
        return xm+x2

class SimpleNN(nn.Module):
    def __init__(self, split):
        super(SimpleNN, self).__init__()
        
        self.nout = 32
        self.repeat = 2
        self.dilation = 1
        self.ks = 3
        self.pad = 1
        self.block_nums = 1
        self.block_nums = block_nums - 1

        GV.repeat = repeat  # HOLD
        
        # OUT
#         self.data, self.labels = L.Python(module = 'read_data1', 
#                                        layer = 'input_layer',
        self.input_layer = InputLayer(
               params = dict(split=split, 
                                    data_dir = '/content/drive/My Drive/MEnet',
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
        self.dilation = set_dilation
        
        class A():    pass
        data=A()
        data.nout = 3

        result = data

        # BRANCH from data
        _, _, _, self.data_f = BN_scale_relu_conv(split, bottom = data, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH
        result_s = []
#         a, b, c, result = conv_BN_scale_relu(split, bottom = result, nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        result_s.append(
            conv_BN_scale_relu(split, bottom = result, nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)[-1]
        )

        for i in range(repeat): #224
            _, _, _, M1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)  
            _, _, _, M2 = BN_scale_relu_conv(split, bottom = M1, nout = nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)        
#             result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
            result_s.append(
                SeqSkip(M1,M2)
            )
        self.stack1 = nn.Sequential(*result_s) # start from data
        self.stack1.nout = self.stack1[-1].nout

        # BRANCH
        scale0 = self.stack1[-1]
        _, _, _, self.scale0_m = BN_scale_relu_conv(split, bottom = scale0, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH
        result_s = []
        for i in range(repeat):#112
            if i == 0:
                _, _, _, result1 = BN_scale_relu_conv(split, bottom = self.stack1, nout = 2 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)  
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
                a, b, c, resultm = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 2 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1, result2, resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1,result2)
                )
        
        # BRANCH
        self.scale1 = nn.Sequential(*result_s)  # from stack1
        self.scale1.nout = self.scale1[-1].nout
        # scale1 = result_s[-1]  # BRANCH

        a, b, c, self.scale1_m = BN_scale_relu_conv(split, bottom = self.scale1, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH
        result_s = []
        for i in range(repeat):#56
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale1, nout = 4 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.scale1, nout = 4 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1, result2)
                )

        self.scale2 = nn.Sequential(*result_s)  # from scale1
        self.scale2.nout = self.scale2[-1].nout

        scale2 = result  # BRANCH
        a, b, c, self.scale2_m = BN_scale_relu_conv(split, bottom = self.scale2, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH
        result_s=[]
        for i in range(repeat):#28
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale2, nout = 8 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.scale2, nout = 8 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1, result2, resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1,result2)
                )
        # BRANCH
        self.scale3 = nn.Sequential(*result_s)  # from scale2
        self.scale3.nout = self.scale3[-1].nout
        # scale3 = result  # BRANCH
        
        # BRANCH
        a, b, c, self.scale3_m = BN_scale_relu_conv(split, bottom = self.scale3, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        result_s = []
        for i in range(repeat):#14
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale3, nout = 16 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.scale3, nout = 16 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1,result2)
                )
        # BRANCH
        self.scale4 = nn.Sequential(*result_s)  # from scale3
        self.scale4.nout = self.scale4[-1].nout

        # scale4 = result  # BRANCH
        a, b, c, self.scale4_m = BN_scale_relu_conv(split, bottom = self.scale4, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        result_s=[]
        for i in range(repeat):#7
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale4, nout = 32 * nout, ks=ks, stride=2, pad = pad, dilation = dilation, in_place = False)
                a, b, c, resultm = BN_scale_relu_conv(split, bottom = self.scale4, nout = 32 * nout, ks=1, stride=2, pad = 0, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 32 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 32 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 32 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1,result2)
                )
        # BRANCH
        self.scale5 = nn.Sequential(*result_s)
        self.scale5.nout = self.scale5[-1].nout
        # scale5 = result
        
        # BRANCH
        a, b, c, self.scale5_m = BN_scale_relu_conv(split, bottom = self.scale5, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        dilation = 1

        # BRANCH
        a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale5, nout = 32 * nout, ks=7, stride=1, pad = 0, dilation = dilation, in_place = False)
        a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 32 * nout, ks =7, stride=1, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
        self.scale5_5 = nn.Sequential(result1,result2)  # fron scale5
        self.scale5_5.nout = self.scale5_5[-1].nout

        a, b, c, self.scale5_u = BN_scale_relu_conv(split, bottom = self.scale5_5, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH TEMPORAL
#         result = L.Concat(self.scale5_5, self.scale5)
        self.scale5_5.nout += self.scale5.nout
        
        result_s = []
        for i in range(repeat):#14
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.scale5_5, nout = 16 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation ,in_place = True)
                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.scale5_5, nout = 16 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation,in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 16 * nout, ks=ks, stride=1, pad = pad, dilation = dilation ,in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1, result2)
                )

        self.stack6=nn.Sequential(*result_s)  # from Concat(self.scale5_5, self.scale5)
        self.stack6.nout = self.stack6[-1].nout
        
        # BRANCH
        a, b, c, self.scale4_u = BN_scale_relu_conv(split, bottom = self.stack6, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH TEMPORAL
#         result = L.Concat(self.stack6, self.scale4)
        self.stack6.nout += self.scale4.nout
        
        # BRANCH
        result_s=[]
        for i in range(repeat):#28
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.stack6, nout = 8 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.stack6, nout = 8 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation,in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 8 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1, result2)
                )
        
        self.stack7=nn.Sequential(*result_s)  # BRANCH
        self.stack7.nout = self.stack7[-1].nout
        
        # BRANCH
        a, b, c, self.scale3_u = BN_scale_relu_conv(split, bottom = self.stack7, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH TEMPORAL
#         result = L.Concat(self.stack7, self.scale3)
        self.stack7.nout += self.scale3
        
        result_s=[]
        for i in range(repeat):#56
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.stack7, nout = 4 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.stack7, nout = 4 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, resultm, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result_s[-1], nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 4 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1,result2)
                )
        self.stack8=nn.Sequential(*result_s)
        self.stack8.nout = self.stack8[-1].nout
        
        a, b, c, self.scale2_u = BN_scale_relu_conv(split, bottom = self.stack8, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH TEMPORAL
#         result = L.Concat(result, self.scale2)
        self.stack8.nout += self.scale2
        
        result_s=[]
        for i in range(repeat):#112
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.stack8, nout = 2 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.stack8, nout = 2 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1,result2,resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 2 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1,result2)
                )
        
        self.stack9=nn.Sequential(*result_s)
        self.stack9.nout=self.stack9[-1].nout

        a, b, c, self.scale1_u = BN_scale_relu_conv(split, bottom = self.stack9, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
        # BRANCH TEMPORAL
#         result = L.Concat(self.stack9, self.scale1)
        self.stack9.nout += self.scale1
        
        result_s = []        
        for i in range(repeat):#224
            if i == 0:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = self.stack9, nout = 1 * nout, ks = 4, stride=2, pad = 1, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, resultm  = BN_scale_relu_conv(split, bottom = self.stack9, nout = 1 * nout, ks = 2, stride=2, pad = 0, dilation = dilation,  conv_type = 'dconv', in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 1 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, resultm, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqMSkip(result1, result2, resultm)
                )
            else:
                a, b, c, result1 = BN_scale_relu_conv(split, bottom = result, nout = 1 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
                a, b, c, result2 = BN_scale_relu_conv(split, bottom = result1, nout = 1 * nout, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = True)
#                 result = L.Eltwise(result2, result, operation = P.Eltwise.SUM)
                result_s.append(
                    SeqSkip(result1,result2)
                )
    
        self.stack10=nn.Sequential(*result_s)
        self.stack10.nout=self.stack10[-1].nout
        
        a, b, c, self.scale0_u = BN_scale_relu_conv(split, bottom = self.stack10, nout = 1, ks=ks, stride=1, pad = pad, dilation = dilation, in_place = False)
        
#         self.resize = L.Python(self.data_f, self.scale0_m, self.scale1_m, self.scale2_m, self.scale3_m, self.scale4_m, self.scale5_m, self.scale0_u, self.scale1_u, self.scale2_u, self.scale3_u, self.scale4_u, self.scale5_u,
#                                        ntop = 1,
#                                        module = 'resize', 
#                                        layer = 'resize_to_same_size',
#                                        param_str = str(dict()))
        self.resize = ResizeToSameSize()
        
        a, b, metric, c = conv_BN_scale_relu(split, bottom = self.resize, nout = 16, ks=3, stride=1, pad = 1, dilation = dilation, in_place = True)
#         metric = L.TanH(metric, in_place= True)
        self.metric = nn.Sequential(metric, nn.Tanh)
    
        a, b, self.sal, d = conv_BN_scale_relu(split, bottom = self.resize, nout = 2, ks=3, stride=1, pad = 1, dilation = dilation, in_place = True)

#         return self.metric, self.sal
        
#         self.cal_loss = L.Python(self.metric, self.sal, self.labels,
#                                        ntop = 1,
#                                        module = 'pyloss', 
#                                        layer = 'EuclideanLossLayer',
#                                        param_str = str(dict(crop_size = [64, 64],
#                                                             batches = 20,
#                                                             sup_threshold = 3,
#                                                             inf_threshold = 0
#                                                            )),loss_weight=1)
#         return to_proto(self.cal_loss)

        def forward(self, data):
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
            
            scale5_5 = torch.cat((scale5_5, scale5), 2)  # guessed C
            
            stack6 = self.stack6(scale5_5)
            scale4_u = self.scale4_u(stack6)
            
            stack6 = torch.cat((stack6,scale4),2)
            
            stack7 = self.stack7(stack6)
            scale3_u = self.scale3_u(stack7)
            stack8 = self.stack8(stack7)
            scale2_u = self.scale2_u(stack8)
            
            stack8 = torch.cat((stack8,scale2),2)
            
            stack9 = self.stack9(stack8)
            scale1_u = self.scale1_u(stack9)
            
            stack9 = torch.cat((stack9,scale1),2)
            
            stack10 = self.stack10(stack9)
            scale0_u = self.scale0_u(stack10)
            
            resize = self.resize(
                data_f, scale0_m, scale1_m, scale2_m, scale3_m, scale4_m,
                scale5_m, scale0_u, scale1_u, scale2_u, scale3_u,
                scale4_u, scale5_u,
            )
            
            metric = self.metric(resize)
            sal = self.sal(resize)
            
            return metric, sal
        
        def forward_withoout_input(self):
#             self.data, self.labels = self.input_layer()
#             data, label = self.data, self.labels
            data, label = self.input_layer()
            self.forward(data)
            

def store_curve_value(snapshot_name, train_dir, test_dir, data_iter = 10,  iter_step = 500):
    loss_file = snapshot_name + '.h5'
    start_index = 0
    if os.path.exists(loss_file):
        print('The file ' + loss_file  + '.h5 exists.')
        h5file = h5py.File(loss_file, 'r')
        train_metric_loss = h5file['train_metric_loss'][...]
        train_sal_loss = h5file['train_sal_loss'][...]
        test_metric_loss = h5file['test_metric_loss'][...]
        test_sal_loss = h5file['test_sal_loss'][...]
        h5file.close()
        start_index = (train_metric_loss.shape[0]) * iter_step
        train_metric_loss = list(train_metric_loss)
        train_sal_loss = list(train_sal_loss)
        test_metric_loss = list(test_metric_loss)
        test_sal_loss = list(test_sal_loss)
        print(start_index)
    else:
        train_metric_loss = []
        train_sal_loss = []
        test_metric_loss = []
        test_sal_loss = []
        print('The file ' + loss_file + ' has set up.')
    for index in range(start_index + iter_step, 110000 + iter_step, iter_step):
#        print(index)
        weight = osp.join('model', 'snapshot', snapshot_name + '_iter_' + str(index) + '.caffemodel')
        train_net = caffe.Net(str(train_dir), str(weight), caffe.TEST)
        test_net = caffe.Net(str(test_dir), str(weight), caffe.TEST)
        tmp_train_metric_loss = 0
        tmp_train_sal_loss = 0
        tmp_test_metric_loss = 0
        tmp_test_sal_loss = 0
        for forward_iter in range(data_iter):
            train_net.forward()
            test_net.forward()
            tmp_train_metric_loss = tmp_train_metric_loss + train_net.blobs['Python4'].data[0]
            tmp_train_sal_loss = tmp_train_sal_loss + train_net.blobs['Python4'].data[1]
            tmp_test_metric_loss = tmp_test_metric_loss + test_net.blobs['Python4'].data[0]
            tmp_test_sal_loss = tmp_test_sal_loss + test_net.blobs['Python4'].data[1]
        train_metric_loss.append(train_metric_loss / data_iter)
        train_sal_loss.append(train_sal_loss / data_iter)
        test_metric_loss.append(test_metric_loss / data_iter)
        test_sal_loss.append(test_sal_loss / data_iter)
#        if index % 20 == 0:
        print('saving loss data', train_metric_loss, train_sal_loss, test_metric_loss, test_sal_loss)
#        da
        h5file = h5py.File(loss_file, 'w')
        h5file.create_dataset('train_metric_loss', data = np.array(train_metric_loss), dtype = np.float32)
        h5file.create_dataset('train_sal_loss', data = np.array(train_sal_loss), dtype = np.float32)
        h5file.create_dataset('test_metric_loss', data = np.array(test_metric_loss), dtype = np.float32)
        h5file.create_dataset('test_sal_loss', data = np.array(test_sal_loss), dtype = np.float32)
        h5file.close()
#    store_curve_value(snapshot_name, train_dir, test_dir)
#    das
def store_curve_to_mat(snapshot_name, train_dir, test_dir, data_iter = 100,  iter_step = 10000):
    snapshot_name = snapshot_name.split('_iter_')[0]
    loss_file = snapshot_name + '.mat'
    start_index = 0
    if os.path.exists(loss_file):
        print('The file ' + loss_file  + '.mat exists.')
        matfile = scio.loadmat(loss_file)
        train_metric_loss = matfile['train_metric_loss']
        train_sal_loss = matfile['train_sal_loss']
        test_metric_loss = matfile['test_metric_loss']
        test_sal_loss = matfile['test_sal_loss']
        start_index = (train_metric_loss.shape[0]) * iter_step
        train_metric_loss = list(train_metric_loss)
        train_sal_loss = list(train_sal_loss)
        test_metric_loss = list(test_metric_loss)
        test_sal_loss = list(test_sal_loss)
        print(start_index)
    else:
        train_metric_loss = []
        train_sal_loss = []
        test_metric_loss = []
        test_sal_loss = []
        print('The file ' + loss_file + ' has set up.')
    for index in range(start_index + iter_step, 110000 + iter_step, iter_step):
#        print(index)
        weight = osp.join('model', 'snapshot', snapshot_name + '_iter_' + str(index) + '.caffemodel')
        train_net = caffe.Net(str(train_dir), str(weight), caffe.TEST)
        test_net = caffe.Net(str(test_dir), str(weight), caffe.TEST)
        tmp_train_metric_loss = 0
        tmp_train_sal_loss = 0
        tmp_test_metric_loss = 0
        tmp_test_sal_loss = 0
        for forward_iter in range(data_iter):
            train_net.forward()
            test_net.forward()
            tmp_train_metric_loss = tmp_train_metric_loss + train_net.blobs['Python4'].data[0]
            tmp_train_sal_loss = tmp_train_sal_loss + train_net.blobs['Python4'].data[1]
            tmp_test_metric_loss = tmp_test_metric_loss + test_net.blobs['Python4'].data[0]
            tmp_test_sal_loss = tmp_test_sal_loss + test_net.blobs['Python4'].data[1]
        train_metric_loss.append(tmp_train_metric_loss / data_iter)
        train_sal_loss.append(tmp_train_sal_loss / data_iter)
        test_metric_loss.append(tmp_test_metric_loss / data_iter)
        test_sal_loss.append(tmp_test_sal_loss / data_iter)
#        if index % 20 == 0:
        print('saving loss data', train_metric_loss, train_sal_loss, test_metric_loss, test_sal_loss)
#        da
        scio.savemat(loss_file, {'train_metric_loss': train_metric_loss,
                                   'train_sal_loss': train_sal_loss,
                                   'test_metric_loss': test_metric_loss,
                                   'test_sal_loss': test_sal_loss})
#        h5file = h5py.File(loss_file, 'w')
#        h5file.create_dataset('train_metric_loss', data = np.array(train_metric_loss), dtype = np.float32)
#        h5file.create_dataset('train_sal_loss', data = np.array(train_sal_loss), dtype = np.float32)
#        h5file.create_dataset('test_metric_loss', data = np.array(test_metric_loss), dtype = np.float32)
#        h5file.create_dataset('test_sal_loss', data = np.array(test_sal_loss), dtype = np.float32)
#        h5file.close()
    
def make_net(snapshot_name = None):
#    this_dir + '/model/train.prototxt'
    if not snapshot_name:
        train_prototxt_dir = './model/train.prototxt'
        test_prototxt_dir = './model/test.prototxt'
    else:
        train_prototxt_dir = './model/train_{}.prototxt'.format(snapshot_name)
        test_prototxt_dir = './model/test_{}.prototxt'.format(snapshot_name)
    with open(train_prototxt_dir, 'w') as f:
        f.write(str(simpleNN('train')))
    with open(test_prototxt_dir, 'w') as f:
        f.write(str(simpleNN('test')))
    solver_dir = './model/solver.prototxt'
    return train_prototxt_dir, test_prototxt_dir, solver_dir
    
GV.hard_samples = []
this_dir = osp.dirname('/content/drive/My Drive/MEnet/training.py')
GV.test_nums = 0
GV.test_images_num = 2
GV.abnormal_files = []

watch_single_result = True

GV.phase = 'train'

# GV.phase = 'test'

GV.test_dir = '/workspace/final_4_DML_model/0305_data_split/'
GV.target_data_dir = '.'

if __name__ == '__main__':
    # make_net()

    train_dir, test_dir, solver_dir = make_net()
    caffe.set_device(0)
    caffe.set_mode_gpu()

    
    #snapshot_name = 'f4_lr_0.1_0408_pyloss_data_sal_split_first5times_pyloss_plus_num_finalout16_two_Loss_ks3_HardNeg_crop_nout32_repeat2_batch5_dilation1_1e-8_iter_110000'
    snapshot_name = 'attention_iter_110000'


    weight = '/content/drive/My Drive/MEnet/model/snapshot/attention/' + str(snapshot_name) + '.caffemodel'

    state = '/content/drive/My Drive/MEnet/model/snapshot/attention' + str(snapshot_name) + '.solverstate'

#     if GV.phase == 'test':
#       net = caffe.Net(str(test_dir), str(weight), caffe.TEST)
# #       net = caffe.Net(str(test_dir), caffe.TEST)
#       net.forward()

#     elif GV.phase == 'train':
#       solver = caffe.SGDSolver(str(solver_dir))
#       solver.step(1)
#       for i in range(1100):
#          solver.step(100)
#     print('done!')


'''''''''
    test_files = os.listdir(osp.join(GV.test_dir))    
    GV.target_data_dir = snapshot_name
    if not os.path.exists(os.path.join(GV.target_data_dir)):
        os.mkdir(os.path.join(GV.target_data_dir))
    for i in range(len(test_files)):
        test_database = os.listdir(osp.join(GV.test_dir, test_files[i]))
        if not os.path.exists(os.path.join(GV.target_data_dir, test_files[i])):
            os.mkdir(os.path.join(GV.target_data_dir, test_files[i]))            
#        sort_key = ['D', 'H', 'E', 'M', 'P', 'S']
#        sort_key = ['D']
#        test_database = sorted(test_database, key= lambda x:sort_key.index(x[0]))
        for j in range(len(test_database)):
            GV.test_data_dir = osp.join(GV.test_dir, test_files[i], test_database[j] + '/')
            GV.test_labels_dir = osp.join(GV.test_dir, test_files[i], test_database[j] + '/')
#            GV.test_labels_dir = osp.join(GV.test_label_dir, test_files[i], test_database[j] + '/')
            if not os.path.exists(os.path.join(GV.target_data_dir, test_files[i], test_database[j])):
                os.mkdir(os.path.join(GV.target_data_dir, test_files[i], test_database[j]))            
            GV.target_dir = osp.join(GV.target_data_dir, test_files[i], test_database[j] + '/')
            train_dir, test_dir, solver_dir = make_net(snapshot_name = snapshot_name)
            net = caffe.Net(str(test_dir), str(weight), caffe.TEST)
#            dsd
            net.forward()
            for p in range(GV.test_images_num):
                net.forward()

'''''''''



    
    
    
    