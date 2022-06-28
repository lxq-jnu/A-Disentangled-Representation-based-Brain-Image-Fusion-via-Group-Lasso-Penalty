from __future__ import print_function
import argparse
import os
import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import copy
from itertools import islice
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
from torchvision import utils
import networks
from fusion_rule import *
from torch.autograd import Variable
from scipy.misc import imread, imsave, imresize





IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.gif'
]

class ImagePair(data.Dataset):
    def __init__(self, impath1, impath2, mode='RGB', transform=None):
        self.impath1 = impath1
        self.impath2 = impath2
        self.mode = mode
        self.transform = transform

    def loader(self, path):
        return Image.open(path).convert(self.mode)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_pair(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

    def get_source(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        return img1, img2


class ImageSequence(data.Dataset):
    def __init__(self, is_folder=False, mode='RGB', transform=None, *impaths):
        self.is_folder = is_folder
        self.mode = mode
        self.transform = transform
        self.impaths = impaths

    def loader(self, path):
        return Image.open(path).convert(self.mode)

    def get_imseq(self):
        if self.is_folder:
            folder_path = self.impaths[0]
            impaths = self.make_dataset(folder_path)
        else:
            impaths = self.impaths

        imseq = []
        for impath in impaths:
            if os.path.exists(impath):
                im = self.loader(impath)
                if self.transform is not None:
                    im = self.transform(im)
                imseq.append(im)
        return imseq

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, img_root):
        images = []
        for root, _, fnames in sorted(os.walk(img_root)):
            for fname in fnames:
                if self.is_image_file(fname):
                    img_path = os.path.join(img_root, fname)
                    images.append(img_path)
        return images



def _crop(img,ow,oh):
    #ow, oh = raw_img.size 
    temp_arr = img[:,:,:oh,:ow]

    return temp_arr

def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
           (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

def load_(net,state_dict):
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        print(state_dict.keys())
        print(state_dict[key].size())
        __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    net.load_state_dict(state_dict)




os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



device = torch.device("cuda:0")

netEnC = networks.define_SimpleEn(netG='C').cuda()

netEnSIR = networks.define_SimpleEn().cuda()
netEnSVI = networks.define_SimpleEn().cuda()

netDe = networks.define_SimpleDe().cuda()



model_EnC_path = "models/MRI-PET/800_net_EnC.pth"

state_dict_EnC = torch.load(model_EnC_path, map_location=str(device))
load_(netEnC,state_dict_EnC)

model_EnSIR_path = "models/MRI-PET/800_net_EnSIR.pth"

state_dict_EnSIR = torch.load(model_EnSIR_path, map_location=str(device))
load_(netEnSIR,state_dict_EnSIR)

model_EnSVI_path = "models/MRI-PET/800_net_EnSVI.pth"

state_dict_EnSVI = torch.load(model_EnSVI_path, map_location=str(device))
load_(netEnSVI,state_dict_EnSVI)

model_netDe_path = "models/MRI-PET/800_net_De.pth"

state_dict_netDe = torch.load(model_netDe_path, map_location=str(device))
load_(netDe,state_dict_netDe)


netEnC.eval()
netEnSIR.eval()
netEnSVI.eval()
netDe.eval()

def fusion_test(real_A, real_B, rule='l1'):
    with torch.no_grad():

        strategy = fusion()



        c_ir = netEnC(real_A)
        c_vi = netEnC(real_B)

        s_ir = netEnSIR(real_A)
        s_vi = netEnSVI(real_B)

        fuse_c = strategy(c_ir, c_vi)

        if rule == 'add':
            fuse_s = s_ir + s_vi
        elif rule == 'l1':
            fuse_s = strategy(s_ir, s_vi)
        elif rule == 'max':
            fuse_s = torch.max(s_ir, s_vi)

        fuse = netDe(fuse_c + fuse_s)

        return fuse



for i in range(0, 10):



    path1 = os.path.join("./test_imgs_mri_pet/",'MRI'+ str(i + 1) + ".png")
    path2 = os.path.join("./test_imgs_mri_pet/", 'PET'+str(i + 1) + ".png")

    img_A = imread(path1, mode='L')
    img_B = imread(path2, mode='YCbCr')

    print(img_A.shape)
    print(img_B.shape)

    y = img_B[:, :, 0]
    cb = np.expand_dims(img_B[:, :, 1], -1)
    cr = np.expand_dims(img_B[:, :, 2], -1)

    # img1 = torch.from_numpy(img_A).float()
    # img2 = torch.from_numpy(y).float()

    img1 = transforms.ToTensor()(img_A)
    img2 = transforms.ToTensor()(y)

    print(img1.size())
    print(img2.size())

    img1.unsqueeze_(0)
    img2.unsqueeze_(0)

    img1 = img1.cuda()
    img2 = img2.cuda()
    with torch.no_grad():

        fuse_Y = fusion_test(img1,img2,rule='add')

    fuse_Y = np.squeeze(fuse_Y.cpu().data[0].numpy() * 255.0, 0).astype('uint8')
    fuse_Y = Image.fromarray(fuse_Y).convert('L')




    image_path = os.path.join("./results_mri_pet/", str(i+1) + ".png")
    fold_path = os.path.join("./results_mri_pet/")
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    imsave(image_path, fuse_Y)




