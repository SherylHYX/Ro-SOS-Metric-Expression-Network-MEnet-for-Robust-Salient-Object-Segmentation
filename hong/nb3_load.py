from nb2_launch import *

# load_state = False

import torch

# LOAD STATE-DICT
def load_state(device):
    print('loading state')
    ckpt_dir = '../tensorboard/pytorch_weights.ckpt'
    ckpt = torch.load(ckpt_dir, map_location=device)

    state=ckpt['state_dict']
    return state

# LOAD FULL
def load_full():
    print('loading full model')
    ckpt_dir = '../ckpt/pytorch_model.ckpt'
    ckpt = torch.load(ckpt_dir)

    # dtype torch.float32
    t2net = ckpt['model']
    data = ckpt['data']
    plabel = ckpt['label']
    tmetric = ckpt['out1_metric']
    tsal = ckpt['out2_sal']
    return t2net, data, plabel, tmetric, tsal
