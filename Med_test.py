import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from itertools import islice
from util import html
import torch
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable

import copy

from scipy.misc import imread, imsave, imresize

import torchvision.models as models_
from torchvision import utils

import time


def ycbcr2rgb(ycbcr_image):
    """convert ycbcr into rgb"""
    if len(ycbcr_image.shape)!=3 or ycbcr_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    ycbcr_image = ycbcr_image.astype(np.float32)
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    transform_matrix_inv = np.linalg.inv(transform_matrix)
    shift_matrix = np.array([16, 128, 128])
    rgb_image = np.zeros(shape=ycbcr_image.shape)
    w, h, _ = ycbcr_image.shape
    for i in range(w):
        for j in range(h):
            rgb_image[i, j, :] = np.dot(transform_matrix_inv, ycbcr_image[i, j, :]) - np.dot(transform_matrix_inv, shift_matrix)
    return rgb_image.astype(np.uint8)


def compute_neighnour_differences(x):
    w = torch.FloatTensor([
        [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, -1]]
    ]).unsqueeze(1)  # 8,1,3,3
    w = w

    # x = torch.norm(torch.abs(x), p=1, dim=1).unsqueeze(1)
    x = x.unsqueeze(0).unsqueeze(0)

    y = torch.nn.functional.conv2d(x, w, padding=1)
    y = y.abs()
    y = y.sum(dim=1, keepdim=True)

    return x, y


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


# vgg = networks.Vgg19().cuda()

# print(models_.vgg19(pretrained=True).features)


def _crop(img, ow, oh):
    # ow, oh = raw_img.size #ow是水平方向，oh是竖直方向
    temp_arr = img[:, :, :oh, :ow]

    return temp_arr


# options
opt = TestOptions().parse()
opt.num_threads = 1  # test code only supports num_threads=1
opt.batch_size = 1  # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
# model.eval()
print('Loading model %s' % opt.model)


# print(type(model.netG.module._modules))


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col



'''
record1=0
for i in range(0, 23):

    path1 = os.path.join("../../../../root/raid1/waq/MRI-CT/MRI/", "MRI" + str(i + 1) + ".png")
    path2 = os.path.join("../../../../root/raid1/waq/MRI-CT/CT/", "CT" + str(i + 1) + ".png")



    img_A = imread(path1, mode='L')
    img_B = imread(path2, mode='YCbCr')


    print(img_A.shape)
    print(img_B.shape)

    y = img_B[:,:,0]
    cb = np.expand_dims(img_B[:, :, 1],-1)
    cr = np.expand_dims(img_B[:, :, 2],-1)

    img1 = transforms.ToTensor()(img_A)
    img2 = transforms.ToTensor()(y)

    print(img1.size())
    print(img2.size())

    img1.unsqueeze_(0)
    img2.unsqueeze_(0)

    img1 = Variable(img1.cuda(),requires_grad=False)
    img2 = Variable(img2.cuda(),requires_grad=False)
    # c1, c2, s1, s2 = model.test2(img1, img2)

    if i+1 == -1:
        c1, c2, s1, s2 = model.test3(img1, img2)

        # x = torch.abs(s1)

        x = torch.norm(s1, p=1, dim=1).squeeze(0)
        x2 = torch.norm(s2, p=1, dim=1).squeeze(0)

        y = torch.max(x, x2)
        z = torch.min(x, x2)



        c1 = c1.permute(1, 0, 2, 3)
        c2 = c2.permute(1, 0, 2, 3)

        s1 = s1.permute(1, 0, 2, 3)
        s2 = s2.permute(1, 0, 2, 3)

        for i in range(0, c1.shape[0]):

            item = s2  # torch.relu((s2+s1))
            item = item.cpu()
            feature_img = item[i][0].data.numpy()
            x_max = np.max(feature_img)

            x_min = np.min(feature_img)

            print(x_max)

            # exit()

            # print(x_min)

            feature_img = (feature_img - x_min) / (x_max - x_min)

            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            web_dir = os.path.join('Visualization', opt.name, "s2")
            if os.path.exists(web_dir) is False:
                os.makedirs(web_dir)

            dst_file = os.path.join(web_dir, str(i) + '.png')
            # dst_file = os.path.join(web_dir, 'rec_cir' + '.png')
            print(dst_file)
            cv2.imwrite(dst_file, feature_img)

        exit()

    #else:
    #    continue


    print(img1.size())
    print(img2.size())

    start = time.time()
    fuse_img = model.fusion_test(img1, img2,'add')
    end = time.time()

    record1 = end-start + record1


    fuse_img = np.squeeze(fuse_img.cpu().data[0].numpy() * 255.0,0).astype('uint8')



    fuse_img=Image.fromarray(fuse_img).convert('L')






    epoch = 100
    # fuse_img = _crop(fuse_img, w, h)
    image_path = os.path.join("./results/MRI_CT/", 'ADD_noreg', str(i+1) + ".png")
    fold_path = os.path.join("./results/MRI_CT/", 'ADD_noreg')
    print(image_path)
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    imsave(image_path,fuse_img)


print("MRI_CT_mean_time",record1/23.0)

'''

#'''
record2=0
for i in range(0, 10):

    path1 = os.path.join("./MRI-test/", "MRI" + str(i + 1) + ".png")
    path2 = os.path.join("./CT-test/", "CT" + str(i + 1) + ".png")



    img_A = imread(path1, mode='L')
    img_B = imread(path2, mode='YCbCr')



    print(img_A.shape)
    print(img_B.shape)

    y = img_B[:,:,0]
    cb = np.expand_dims(img_B[:, :, 1],-1)
    cr = np.expand_dims(img_B[:, :, 2],-1)


    #img1 = torch.from_numpy(img_A).float()
    #img2 = torch.from_numpy(y).float()

    img1 = transforms.ToTensor()(img_A)
    img2 = transforms.ToTensor()(y)

    print(img1.size())
    print(img2.size())

    img1.unsqueeze_(0)
    img2.unsqueeze_(0)

    img1 = Variable(img1.cuda(),requires_grad=False)
    img2 = Variable(img2.cuda(),requires_grad=False)
    # c1, c2, s1, s2 = model.test2(img1, img2)

    if i+1 == -1:
        c1, c2, s1, s2 = model.test3(img1, img2)

        # x = torch.abs(s1)

        x = torch.norm(s1, p=1, dim=1).squeeze(0)
        x2 = torch.norm(s2, p=1, dim=1).squeeze(0)

        y = torch.max(x, x2)
        z = torch.min(x, x2)



        c1 = c1.permute(1, 0, 2, 3)
        c2 = c2.permute(1, 0, 2, 3)

        s1 = s1.permute(1, 0, 2, 3)
        s2 = s2.permute(1, 0, 2, 3)

        for i in range(0, c1.shape[0]):

            item = s2  # torch.relu((s2+s1))
            item = item.cpu()
            feature_img = item[i][0].data.numpy()
            x_max = np.max(feature_img)

            x_min = np.min(feature_img)

            print(x_max)

            # exit()

            # print(x_min)

            feature_img = (feature_img - x_min) / (x_max - x_min)

            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            web_dir = os.path.join('Visualization', opt.name, "s2")
            if os.path.exists(web_dir) is False:
                os.makedirs(web_dir)

            dst_file = os.path.join(web_dir, str(i) + '.png')
            # dst_file = os.path.join(web_dir, 'rec_cir' + '.png')
            print(dst_file)
            cv2.imwrite(dst_file, feature_img)

        exit()

    #else:
    #    continue


    print(img1.size())
    print(img2.size())

    start = time.time()
    fuse_img = model.fusion_test(img1, img2,'add')
    end = time.time()

    record2 = end-start + record2


    fuse_img = np.squeeze(fuse_img.cpu().data[0].numpy() * 255.0,0).astype('uint8')



    fuse_img=Image.fromarray(fuse_img).convert('L')






    epoch = 100
    # fuse_img = _crop(fuse_img, w, h)
    image_path = os.path.join("./results/MRI_CT/", 'ADD', str(i+1) + ".png")
    fold_path = os.path.join("./results/MRI_CT/", 'ADD')
    print(image_path)
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    imsave(image_path,fuse_img)


print("MRI_PET_mean_time",record2/27.0)
#'''