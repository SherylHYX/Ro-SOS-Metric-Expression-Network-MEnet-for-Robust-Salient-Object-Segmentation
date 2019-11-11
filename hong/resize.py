# import caffe
# import numpy as np
# import global_var as GV
# from skimage.transform import resize as imresize
# import random
# import matplotlib.pyplot as plt
# from PIL import Image
from pdb import set_trace
import torch
from torch import nn


# class resize_to_same_size(caffe.Layer):
class ResizeToSameSize(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(ResizeToSameSize, self).__init__()
        self.device = device
        self.dtype = dtype
        self.shape_dict = {  # without 1,7
            '2': (1,1,352,352),
            '3': (1,1,352,352),
            '4': (1,1,352,352),
            '5': (1,1,352,352),
            '6': (1,1,352,352),

            '8': (1,1,352,352),
            '9': (1,1,352,352),
            '10': (1,1,352,352),
            '11': (1,1,352,352),
            '12': (1,1,352,352)
        }
        self.data_dict = {}
        self.num_in = 12
        self.make()  # fill in data_dict before runtime
        # def clear_up(f):
        #     def new_f(*args,**key_args):
        #         result = f(*args, **key_args)
        #         for ten in self.data_dict.values():
        #             ten.detach_()
        #             self.zero_grad()
        #         return result
        #     return new_f
        # self.backward = clear_up(self.backward)

    def make(self):
        for i in range(1,self.num_in+1):
            if str(i) in self.shape_dict.keys():
                shape = self.shape_dict[str(i)]
                self.data_dict[str(i)] = torch.zeros([shape[0], shape[1], shape[2], shape[3]], dtype = self.dtype, device=self.device)

    def clear(self):
        for v in self.data_dict.values():
            v.detach_()
    
    def simple_resize(self, data, i, target_size):
        original_size = data.shape
#         resize_data = np.zeros([original_size[0], original_size[1], target_size[2], target_size[3]])
#         resize_data = torch.zeros([original_size[0], original_size[1], target_size[2], target_size[3]], dtype = data.dtype, device=self.device)

        resize_data = self.data_dict[str(i)]
        resize_data[...] = 0

        # print('\'{}\':({},{},{},{}),'.format(i, original_size[0], original_size[1], target_size[2], target_size[3]))

        x_step = int(target_size[2] // original_size[2])
        y_step = int(target_size[3] // original_size[3])
        for i in range(original_size[2]):
            for j in range(original_size[3]):
                resize_data[:, :, i * x_step : (i + 1)* x_step, j * y_step : (j + 1) * y_step] = data[:, :, i, j]  # HONG modified , np.newaxis, np.newaxis]
        return resize_data

#     def calcu_diff(self, total_diff, ID, size_record):
#         x_step = int(size_record[0][2] // size_record[ID][2])
#         y_step = int(size_record[0][3] // size_record[ID][3])
#         channel_start = 0
#         for i in range(ID):
#             channel_start += size_record[i][1]
            
#         if size_record[ID] == size_record[0]:
#                 diff = total_diff[:, channel_start : channel_start + size_record[ID][1]]
#         else:
#             diff = np.zeros(size_record[ID])
#             for i in range(size_record[ID][2]):
#                 for j in range(size_record[ID][3]):
#                     diff[:, :, i, j] = np.sum(total_diff[:, channel_start : channel_start + size_record[ID][1], i * x_step : (i + 1) * x_step, j * y_step : (j + 1) * y_step].reshape(size_record[ID][0], size_record[ID][1], x_step * y_step), axis = 2)

#         return diff
    
#     def reshape(self, bottom, top):
    def forward(self, *bottom):
        self.N = len(bottom)
        tagert_size = bottom[0].shape
        self.data = bottom[0]
        self.size_record = [tagert_size]
        
        datum = [self.data]
        
        for i in range(1, self.N):
            if bottom[i].shape != tagert_size:  # i=
                resized_data = self.simple_resize(bottom[i], i, tagert_size)
            else:  # i=1,7
                # print(i)
                resized_data = bottom[i]
            self.size_record.append(bottom[i].shape)
            datum.append(resized_data)
#             self.data = torch.cat((self.data, resized_data), axis = 1)

        # set_trace()
        # datum = [data.to(self.device) for data in datum]
        self.data = torch.cat(datum, axis=1)
#         top[0].reshape(*self.data.shape)      
        return self.data
        
#     def backward(self, top, propagate_down, bottom):
#         GV.record_diff = []
#         for i in range(self.N):
#             bottom[i].diff[...] = self.calcu_diff(top[0].diff, i, self.size_record)
#             GV.record_diff.append(self.calcu_diff(top[0].diff, i, self.size_record))
#             GV.all_diff = top[0].diff
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        