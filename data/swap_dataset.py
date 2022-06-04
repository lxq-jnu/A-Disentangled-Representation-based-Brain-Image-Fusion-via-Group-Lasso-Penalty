import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import random
import numpy as np
from random import choice
import matplotlib.cm as mtpltcm
import matplotlib as mpl
from PIL import Image, ImageEnhance


# 用于生成类似红外和可见光的数据集
# 使用的数据为源图像和label图像
class SWAPDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, 'test8_rgb','nyu_ir')
        self.dir_B = os.path.join(opt.dataroot, 'test8_rgb','nyu_vi')
        self.dir_F = os.path.join(opt.dataroot, 'nyu_mask')
        print("HI!!!!!!!!!!!!!!!!")
        print(self.dir_F)
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.F_paths = sorted(make_dataset(self.dir_F, opt.max_dataset_size))
        print(len(self.A_paths))
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - Ground Truth
            B (tensor) - - Train_d
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        A_path = self.A_paths[index]  # A是原图
        B_path = self.B_paths[index]  # D是深度图
        F_path = self.F_paths[index]  # L是分割
        # AB = Image.open(AB_path).convert('RGB')

        ir = load_label_image(A_path)
        vis = load_label_image(B_path)
        gt = load_label_image(F_path)

        # apply the same transform to both A and B and F
        transform_params = get_params(self.opt, ir.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        F_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        IR = A_transform(ir)
        VIS = B_transform(vis)
        M = F_transform(gt)

        return {'A': IR, 'B': VIS, 'M': M, 'A_paths': A_path, 'B_paths': A_path, 'F_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)


def load_label_image(path):
    """Loads an unprocessed Train_d map extracted from the raw dataset."""
    return Image.open(path).convert('L')

