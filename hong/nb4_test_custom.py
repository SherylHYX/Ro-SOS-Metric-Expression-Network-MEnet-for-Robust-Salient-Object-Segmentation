import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from nb2_launch import *
from nb3_load import *
import time



# read data:

# ----- CELL 1 -----

from skimage.transform import resize as imresize
from pdb import set_trace
from datasets import d_dict


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, name, dtype, device, resize_size=(352,352)):
        self.filenames = d_dict[name]
        # self.resize_size=(224,224)
        self.resize_size = resize_size  # 352-224 224-176
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data_dir = self.filenames[idx]
        suffix = data_dir.split('.')[-1]  # suffix = osp.join(data_dir,  self.test_data_dir, self.test_images[self.test_timing]).split('.')[-1]
        if suffix == 'png':
            image_data = mping.imread(data_dir) * 255
        else:
            image_data = mping.imread(data_dir)
        # try:
        #     image_labels = mping.imread(self.test_labels_dir)
        # except:
        #     image_labels = mping.imread(self.test_labels_dir)

        # image_labels.flags.writeable = True  # Hong_add_this
        # image_labels[np.where(image_labels>0.1)] = 1

        # print(data_dir, self.test_labels_dir, self.test_images[self.test_timing], GV.data_name)
        if len(image_data.shape) == 3:
            if image_data.shape[2] != 3:
                # GV.image = image_data
                image_data = imresize(np.array(image_data[:,:,0] * 255, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
                image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
            else:
                image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1], 3])
        elif len(image_data.shape) == 2:
            image_data = imresize(np.array(image_data, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
            image_data = np.tile(image_data[:,:,np.newaxis], [1,1,3])
        # set_trace()
        # if len(image_labels.shape) == 3:
        #     # image_labels = imresize(np.array(image_labels[:,:,0], dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
        #     image_labels = imresize(image_labels[:,:,0], [self.resize_size[0], self.resize_size[1]])
        # else:
        #     # image_labels = imresize(np.array(image_labels, dtype = np.uint8), [self.resize_size[0], self.resize_size[1]])
        #     image_labels = imresize(image_labels, [self.resize_size[0], self.resize_size[1]])
        # set_trace()
        image_data = np.uint8(np.round(image_data*255))
        # GV.image = image_data
        return image_data.transpose(2, 0, 1)
            # self.data[i] = image_data.transpose(2, 0, 1)
            # self.labels[i, 0] = image_labels

# (self.data, self.labels)
# set_trace()



# MAKE NET
device = torch.device('cuda:1')
dtype = torch.float32

ds = TestDataset('HKU_IS', dtype=dtype,device=device)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
# x = torch.tensor(self.data,dtype=dtype,requires_grad=True, device=device)

state=load_state(device)
t2net = SimpleNN(device,dtype=dtype)
t2net=t2net.to(device)

t2net.load_state_dict(state_dict=state)

# from PIL import Image
# img = np.uint8(self.data[0].transpose(1,2,0))
# Image.fromarray(img).save('../'+data_dir.split('/')[-3]+'/'+data_dir.split('/')[-1])
# plt.imshow(img)
# set_trace()

# ------CELL 2-------
softmax = nn.Softmax(dim=1,)
def soft(test):
    # test[test<1e-10]=1e-10  # clamp
    return test

# ------CELL 3-------
# x(0-255) --model--> tsal --Softmax--> sm  --C=1-->  sm  --min_max-->  final saliency map
# sm=soft(tsal)  # Softmax: B=1,C,H,W


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.t2net = t2net
        self.softmax = softmax

    def forward(self,x):
        _, tsal = self.t2net(x)
        tsal = torch.nn.functional.interpolate(tsal, size=(176, 176), scale_factor=None, mode='nearest',
                                               align_corners=None)
        # sm = soft(tsal)
        test = tsal
        test = test - torch.max(test, dim=1, keepdim=True)[0]
        test = softmax(test)
        sm = test.sum()
        return sm


testnet = TestNet()

# tmetric, tsal = t2net.forward(x)
# tsal=torch.nn.functional.interpolate(tsal, size=(176,176), scale_factor=None, mode='nearest', align_corners=None)

# sm=sm[:,1,:,:].view(sm.shape[0],-1) # B=1,HW

# EPSILON=1e-8
# sm = (sm - sm.min(dim=1,keepdim=True)[0] + EPSILON) / (sm.max(dim=1,keepdim=True)[0] - sm.min(dim=1,keepdim=True)[0] + EPSILON) * 255

# x = torch.tensor(next(iter(dl)),dtype=dtype,requires_grad=True, device=device)

for i,d in enumerate(dl):
    # t = torch.Tensor()
    # x = t.new_tensor(d, dtype=dtype, requires_grad=True, device=device)
    # x = torch.tensor(d,dtype=dtype,device=device,requires_grad=True)
    x = d.type(dtype).to(device).clone().detach().requires_grad_(True)
    testnet.zero_grad()
    # backend.zero_grad()
    # x=x.to(device)
    start = time.time()
    backend=testnet(x)
    forw = time.time()
    print('!')
    backend.backward()

    backw = time.time()
    g = x.grad.data.clone()

    testnet.t2net.clear()

    # x.detach_()
    print('calculated, forw:{}, backw:{}'.format(forw-start, backw-forw))
# except:
#     print('oops')
# finally:
    print(g.shape)
    print(g.abs().max().item(), g.abs().min().item(), g.abs().median().item(), g.abs().mean().item(), g.abs().var().item())

    


