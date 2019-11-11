# import caffe
import scipy.io as scio
import os.path as osp
import h5py
import numpy as np
import random
# import read_binaryproto
# import read_lmdb
import matplotlib.pyplot as plt
import matplotlib.image as mping
from PIL import Image
import os
import global_var as GV
# from scipy.misc import imresize
from skimage.transform import resize as imresize

import torch
from torch import nn


# class input_layer(caffe.Layer):
class InputLayer(nn.Module):
    def __init__(self, params={}):
        super(InputLayer, self).__init__()
#         params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.split = params['split']
        self.train_data_dir = params['train_data_dir']
        self.train_sobel_dir = params['train_sobel_dir']
        self.train_labels_dir = params['train_labels_dir']
        self.test_labels_dir = params['test_labels_dir']
        self.test_sobel_dir = params['test_sobel_dir']
        self.test_data_dir = params['test_data_dir']
        self.batch_size = params['batch_size']
        self.test_batch_size = params['test_batch_size']
        self.resize_size = params['resize_size']
        self.crop_ratio = 0.9

        self.num_classes = 2
        self.train_timing = 0
        self.test_timing = 0
        self.train_images = os.listdir(osp.join(self.data_dir, self.train_data_dir))
        self.test_images = os.listdir(osp.join(self.data_dir, self.test_data_dir))
        GV.test_images = self.test_images
        self.train_images_num = self.train_images.__len__()
        self.test_images_num = self.test_images.__len__()
        GV.test_images_num = self.test_images_num
        GV.normal_training = 1
        
    def forward(self):
#     def reshape(self, bottom, top):
        if self.split == 'train':
            self.data = np.zeros([self.batch_size, 3, self.resize_size[0], self.resize_size[1]])
            self.labels = np.zeros([self.batch_size, 1, self.resize_size[0], self.resize_size[1]])
            for i in range(self.batch_size):
                self.train_timing = (self.train_timing + 1) % self.train_images_num
                orignial_image_data = mping.imread(osp.join(self.data_dir,  self.train_data_dir, self.train_images[self.train_timing]))
                orignial_image_labels = mping.imread(osp.join(self.data_dir, self.train_labels_dir, self.train_images[self.train_timing].split('.jpg')[0] + '.png'))
                
                if orignial_image_data.shape[:2] != orignial_image_labels.shape[:2]:
                    raise Exception('image and labels must be same size')
                height, width = orignial_image_data.shape[:2]
                
                tmp_crop_ratio = random.randint(int(100 * self.crop_ratio), 100) / 100.
                end_x = random.randint(int(height * tmp_crop_ratio), height)
                end_y = random.randint(int(width * tmp_crop_ratio), width)
                image_data = orignial_image_data[end_x - int(height * tmp_crop_ratio) : end_x, end_y - int(width * tmp_crop_ratio) : end_y]
                image_labels = orignial_image_labels[end_x - int(height * tmp_crop_ratio) : end_x, end_y - int(width * tmp_crop_ratio) : end_y]
                flip = np.random.randint(0, 2)
                if len(image_data.shape) == 3:
                    image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1], 3])
                else:
                    image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                    image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
                    GV.abnormal_files.append(GV.data_name)
                if len(image_labels.shape) == 3:
                    image_labels = imresize(np.array(image_labels[:,:,0], dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                    GV.abnormal_files.append(GV.data_name)
                else:
                    image_labels = imresize(np.array(image_labels, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                image_labels[np.where(image_labels>0)] = 1
                if flip == 1:
                    image_data = np.fliplr(image_data)
                    image_labels = np.fliplr(image_labels)
                self.data[i] = image_data.transpose(2, 0, 1)
                self.labels[i, 0] = image_labels
            
        elif self.split == 'test':
            self.data = np.zeros([self.test_batch_size, 3, self.resize_size[0], self.resize_size[1]])
            self.labels = np.zeros([self.test_batch_size, 1, self.resize_size[0], self.resize_size[1]])
            for i in range(self.test_batch_size):
                self.test_timing = (self.test_timing + 1) % self.test_images_num
                suffix = osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing]).split('.')[-1]
                if suffix == 'png':
                    image_data = mping.imread(osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing])) * 255
                else:
                    image_data = mping.imread(osp.join(self.data_dir,  self.test_data_dir, self.test_images[self.test_timing]))
                try:
                    image_labels = mping.imread(osp.join(self.data_dir, self.test_labels_dir, self.test_images[self.test_timing].split('.' + suffix)[0] + '.png'))
                except:
                    image_labels = mping.imread(osp.join(self.data_dir, self.test_labels_dir, self.test_images[self.test_timing]))

                image_labels.flags.writeable = True  # Hong_add_this
                image_labels[np.where(image_labels>0.1)] = 1
                
                GV.data_name = self.test_images[self.test_timing].split('.')[0]
                GV.data_dir = osp.join(self.data_dir,  self.test_data_dir)
                print(self.data_dir, self.test_labels_dir, self.test_images[self.test_timing], GV.data_name)
                if len(image_data.shape) == 3:
                    if image_data.shape[2] != 3:
                        GV.image = image_data
                        image_data = imresize(np.array(image_data[:,:,0] * 255, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                        image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
                    else:
                        image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1], 3])
                elif len(image_data.shape) == 2:
                    image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                    image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
                
                if len(image_labels.shape) == 3:
                    image_labels = imresize(np.array(image_labels[:,:,0], dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                else:
                    image_labels = imresize(np.array(image_labels, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                image_data = np.uint8(np.round(image_data*255))
                GV.image = image_data
                self.data[i] = image_data.transpose(2, 0, 1)
                self.labels[i, 0] = image_labels

        return self.data, self.labels
#         top[0].reshape(*self.data.shape)
#         top[1].reshape(*self.labels.shape)
        
        
#         top[0].data[...] = self.data
#         top[1].data[...] = self.labels
