"""General-purpose training script for image-to-image translation.
This script works for various models (with option '--model': e.g., bicycle_gan, pix2pix, test) and
different datasets (with option '--dataset_mode': e.g., aligned, single).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
Example:
    Train a BiCycleGAN model:
        python train.py --dataroot ./datasets/facades --name facades_bicyclegan --model bicycle_gan --direction BtoA
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/train_options.py for more training options.
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from data.image_folder import ImageFolder,make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
from util.util import tensor2im,save_image
import os

import models.networks_V2 as networks

import torch

import copy
from itertools import islice
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
from torchvision import utils


def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message


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
    #ow, oh = raw_img.size #ow是水平方向，oh是竖直方向
    temp_arr = img[:,:,:oh,:ow]

    return temp_arr


if __name__ == '__main__':


    opt = TrainOptions().parse()   # get training options

    '''加载测试数据集'''
    test_opt = copy.copy(opt)
    test_opt.preprocess = 'pad'
    #test_opt.preprocess = "scale_width"
    #test_opt.preprocess = "no"
    test_opt.no_flip = True
    test_opt.dataroot = "../../../root/raid1/waq/RoadScene44/"
    test_opt.dataset_mode = "padcrop"
    #test_opt.dataset_mode = "aligned2"
    test_opt.gpu_ids = 1
    test_opt.phase = "test"
    #test_opt.load_size = 512
    test_opt.num_threads = 0
    test_opt.batch_size = 1  # test code only supports batch_size=1
    test_opt.serial_batches = True  # no shuffle
    #test_opt.results_dir = "../results/fm_test92"
    dataset_test = create_dataset(test_opt)  # create a dataset given opt.dataset_mode and other options

    '''加载测试数据集'''

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    total_iters = 0                # the total number of training iterations


    for name in model.model_names:
        if isinstance(name, str):
            net = getattr(model, 'net' + name)
            net.train()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            if not model.is_train():      # if this batch of input data is enough for training.
                print('skip this batch')
                continue
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights



            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)


            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.


        if (epoch)% 5 == 0:  # cache our model every <save_epoch_freq> epochs
            #model.eval()
            start = time.time()
            for i in range(0, 27):
                # path1= os.path.join("../../../../root/raid1/waq/A/",str(i+1)+".jpg")
                # path2 = os.path.join("../../../../root/raid1/waq/B/", str(i+1) + ".jpg")

                path1= os.path.join("../MRI-PET/MRI/","MRI"+str(i+1)+".png")
                path2= os.path.join("../MRI-PET/PET/","PET"+str(i+1)+".png")


                img_A = Image.open(path1).convert('L')


                w, h = img_A.size

                print(w)
                print(h)

                pair_loader = ImagePair(impath1=path1, impath2=path2,
                                        transform=transforms.Compose([
                                            transforms.Grayscale(1),
                                            #transforms.Resize((256,256)),
                                            #transforms.Pad((0, 0, 1024 - w, 1024 - h), padding_mode="reflect"),
                                            transforms.ToTensor(),
                                            #transforms.Normalize(mean=(0.5,), std=(0.5,))

                                        ]))
                img1, img2 = pair_loader.get_pair()
                img1.unsqueeze_(0)
                img2.unsqueeze_(0)

                img1 = img1.cuda()
                img2 = img2.cuda()
                _, fuse_img, _ = model.test2(img1, img2)
                #fuse_img = _crop(fuse_img, w, h)
                image_path = os.path.join("../results/", opt.name, str(epoch), str(i) + ".png")
                fold_path = os.path.join("../results/", opt.name, str(epoch))
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)
                #utils.save_image(
                #    fuse_img, image_path, nrow=fuse_img.shape[0] + 1, normalize=True,
                #    range=(-1, 1)
                #)

                utils.save_image(
                    fuse_img, image_path, nrow=fuse_img.shape[0] + 1, normalize=True,
                    range=(0, 1)
                )
            end = time.time()
            print("用时：",start-end)

            for name in model.model_names:
                if isinstance(name, str):
                    net = getattr(model, 'net' + name)
                    net.train()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
