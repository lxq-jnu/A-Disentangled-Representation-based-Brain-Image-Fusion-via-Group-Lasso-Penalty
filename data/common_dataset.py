import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class CommonDataset(BaseDataset):
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
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot,'train8_ir')  # get the image directory
        self.dir_B = os.path.join(opt.dataroot, 'train8_vi')
        self.dir_F = os.path.join(opt.dataroot, 'train8_f',)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.F_paths = sorted(make_dataset(self.dir_F, opt.max_dataset_size))
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        # self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.input_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        F_path = self.F_paths[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        F = Image.open(F_path).convert('RGB')
        w, h = A.size

        # split AB image into A and B

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        '''
        if self.opt.phase == "test":#如果测试图片是灰度图，复制成三个通道再处理
            if self.opt.gray: 
                A_transform = get_transform(self.opt, transform_params, grayscale=True)
                B_transform = get_transform(self.opt, transform_params, grayscale=True)
                A = A_transform(A)
                B = B_transform(B)
                A = A.repeat(3,1,1)
                B = B.repeat(3,1,1)
        '''

        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        F_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        F = F_transform(F)

        return {'A': A, 'B': B,'F':F, 'A_paths': A_path, 'B_paths': B_path, 'F_path':F_path, 'w': w, 'h': h}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
