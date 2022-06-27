import torch
from .base_model import BaseModel
import networks
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

from fusion_rule import *


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
    w = w.cuda()

    #x = torch.norm(torch.abs(x), p=1, dim=1).unsqueeze(1)

    y = torch.nn.functional.conv2d(x, w, padding=1)
    y = y.abs()
    y = y.sum(dim=1, keepdim=True)

    return y


class MedModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='unet_256', dataset_mode='yuv')
        parser.set_defaults(where_add='input', nz=0)
        if is_train:
            parser.set_defaults(gan_mode='lsgan', lambda_l1=100.0)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.train_mode = "normal"
        BaseModel.__init__(self, opt)

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['EnC', 'EnSIR', 'EnSVI', 'De','DIR','DVI']
        else:  # during test time, only load G
            self.model_names = ['EnC', 'EnSIR', 'EnSVI', 'De']
        # define networks (both generator and discriminator)

        self.strategy = fusion()

        self.netEnC = networks.define_SimpleEn(netG='C')

        self.netEnSIR = networks.define_SimpleEn()
        self.netEnSVI = networks.define_SimpleEn()


        self.netDe = networks.define_SimpleDe()


        self.lap = LaplacianConv()

        if self.isTrain:
            # define loss functions



            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionSSIM = networks.SSIM()  
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()
            self.criterionKLD = torch.nn.KLDivLoss()
            #self.lap = LaplacianConv()

            self.feature_sparsity = Regularization()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_EnC = torch.optim.Adam(self.netEnC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_EnSIR = torch.optim.Adam(self.netEnSIR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_EnSVI = torch.optim.Adam(self.netEnSVI.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_De = torch.optim.Adam(self.netDe.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_EnC)
            self.optimizers.append(self.optimizer_EnSIR)
            self.optimizers.append(self.optimizer_EnSVI)
            self.optimizers.append(self.optimizer_De)





    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if 'F' in input.keys():
            self.real_F = input['F'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']



    def fusion_test(self, real_A, real_B,rule='l1'):
        with torch.no_grad():
            # concat_AB = torch.cat([self.real_A,self.real_B],1)
            self.strategy = fusion()
            self.strategy2 = test_F()

            self.real_A = real_A
            self.real_B = real_B


            self.c_ir = self.netEnC(self.real_A)
            self.c_vi = self.netEnC(self.real_B)

            self.s_ir = self.netEnSIR(self.real_A)
            self.s_vi = self.netEnSVI(self.real_B)




            fuse_c = self.strategy(self.c_ir ,self.c_vi)

            if rule == 'add':
                fuse_s = self.s_ir + self.s_vi
            elif rule == 'l1':
                fuse_s =  self.strategy(self.s_ir ,self.s_vi)
            elif rule == 'max':
                fuse_s = torch.max(self.s_ir,self.s_vi)
       

            self.fuse = self.netDe(fuse_c+fuse_s)
            # print(self.c_ir)

            return self.fuse

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # self.c_ir, self.mu_ir, self.var_ir = self.netEnC(self.real_A)
        # self.c_vi, self.mu_vi, self.var_vi = self.netEnC(self.real_B)

        self.strategy2 = test_F()

        self.grad_ir = self.lap(self.real_A)
        self.grad_vi = self.lap(self.real_B)


        self.c_ir = self.netEnC(self.real_A)
        self.c_vi = self.netEnC(self.real_B)



        self.s_ir = self.netEnSIR(self.real_A)
        self.s_vi = self.netEnSVI(self.real_B)

        self.fake_CA = self.netDe(self.c_ir)
        self.fake_CB = self.netDe(self.c_vi)

        self.fake_SA = self.netDe(self.s_ir)
        self.fake_SB = self.netDe(self.s_vi)

        self.rec_A = self.netDe(self.c_ir + self.s_ir)
        self.rec_B = self.netDe(self.c_vi + self.s_vi)

        self.fake_A = self.netDe(self.c_vi + self.s_ir)
        self.fake_B = self.netDe(self.c_ir + self.s_vi)





    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_L1 = (self.criterionL1(self.rec_A, self.real_A) + self.criterionL1(self.rec_B, self.real_B)) * self.opt.lambda_L1\
                          +(self.criterionL1(self.fake_A, self.real_A) + self.criterionL1(self.fake_B, self.real_B)) * self.opt.lambda_L1\
                         +self.criterionL1(self.fake_CA, self.fake_CB)



        self.loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            cat_fake1 = torch.cat([self.rec_A, self.rec_A, self.rec_A], 1)
            cat_real1 = torch.cat([self.real_A, self.real_A, self.real_A], 1)
            cat_fake2 = torch.cat([self.rec_B, self.rec_B, self.rec_B], 1)
            cat_real2 = torch.cat([self.real_B, self.real_B, self.real_B], 1)

            cat_fake3 = torch.cat([self.fake_A, self.fake_A, self.fake_A], 1)
            cat_fake4 = torch.cat([self.fake_B, self.fake_B, self.fake_B], 1)
            self.loss_G_VGG = (self.criterionVGG(cat_fake1, cat_real1) + self.criterionVGG(cat_fake2, cat_real2)) * self.opt.lambda_feat\
                            +(self.criterionVGG(cat_fake3, cat_real1) + self.criterionVGG(cat_fake4, cat_real2)) * self.opt.lambda_feat

        self.loss_z_L1 = 0

        if self.opt.lambda_z > 0.0:

            self.loss_z_L1 = self.feature_sparsity(self.s_ir,self.s_vi)*self.opt.lambda_z





        self.loss_SSIM = 0
        if self.opt.lambda_SSIM > 0.0:
            self.loss_SSIM = (1 - self.criterionSSIM(self.rec_A, self.real_A)) * self.opt.lambda_SSIM\
                             +(1 - self.criterionSSIM(self.rec_B, self.real_B)) * self.opt.lambda_SSIM\
                                +(1 - self.criterionSSIM(self.fake_B, self.real_B)) * self.opt.lambda_SSIM*0\
                             + (1 - self.criterionSSIM(self.fake_A, self.real_A)) * self.opt.lambda_SSIM*0
                #self.loss_SSIM = self.criterionSSIM_mod(self.real_A, self.real_B, self.fake_F) * self.opt.lambda_SSIM



        self.loss_grad = 0

        if self.opt.lambda_grad >0.0:

            self.loss_grad =(self.criterionL1(self.lap(self.rec_A), self.grad_ir) + self.criterionL1(self.lap(self.rec_B), self.grad_ir)) * self.opt.lambda_grad+\
            (self.criterionL1(self.lap(self.fake_A), self.grad_ir) * self.opt.lambda_grad*0 + self.criterionL1(self.lap(self.fake_B), self.grad_ir)) * self.opt.lambda_grad*0



        self.loss_G =  self.loss_G_L1 + self.loss_G_VGG + self.loss_SSIM + self.loss_grad + self.loss_z_L1
        # combine loss and calculate gradients

        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)

        self.backward_G()  # calculate graidents for G
        self.optimizer_EnC.step()  # udpate G's weights
        self.optimizer_EnSIR.step()
        self.optimizer_EnSVI.step()
        self.optimizer_De.step()

        self.optimizer_EnC.zero_grad()  # set G's gradients to zero
        self.optimizer_EnSIR.zero_grad()  # set G's gradients to zero
        self.optimizer_EnSVI.zero_grad()  # set G's gradients to zero
        self.optimizer_De.zero_grad()



class SobelConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, img):
        self.filter.weight.data = self.filter.weight.data.to(img.device)

        x = self.pad(img)

        x = self.filter(x)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class LaplacianConv(nn.Module):

    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  
        kernel = np.repeat(kernel, self.channels, axis=0)  
        #self.pad = nn.ReflectionPad2d(1)
        self.pad = nn.ReplicationPad2d(1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)  

    def __call__(self, x):
      
        self.weight.data = self.weight.data.to(x.device)
        x = self.pad(x)

        x = F.conv2d(x, self.weight, padding=0, groups=self.channels)
        return x


class Regularization(nn.Module):
    def __init__(self):
        '''
        :param model: model
        :param weight_decay: Regularization parameters
        '''
        super(Regularization, self).__init__()

    def to(self, device):
        '''
        Specify operating mode
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, ir,vi):

        reg_loss = self.regularization_loss5(ir, vi)
    

        return reg_loss


    def smooth(self, tensor, a = 0.02):

        temp_norm = torch.norm(tensor,p=2,dim=1)
        map_l = (temp_norm >=a).float()
        map_s = (temp_norm < a).float()

        comb = map_l*temp_norm + map_s *(0.5*torch.pow(temp_norm,2)/a + a*0.5)

        return comb

    def r(self,x,y):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(x,y) #(1,128,128)

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def tv(self,x):
        batch_size = x.size()[0]

        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        #h_dep = self.r(x[:, :, 1:, :],x[:, :, :h_x - 1, :])
        #w_dep = self.r(x[:, :, :, 1:], x[:, :, :, :w_x - 1])

        h_tv = self.smooth(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        w_tv = self.smooth(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])

        #h_norm = torch.norm(torch.pow(h_dep,4)*h_tv,p=1)
        #w_norm = torch.norm(torch.pow(w_dep,4)*w_tv,p=1)

        h_norm = torch.norm(h_tv,p=1)
        w_norm = torch.norm(w_tv,p=1)

        return (h_norm/count_h+w_norm/count_w)/batch_size




    def regularization_loss5(self, ir, vi):
        '''
        Compute tensor norm
        :param weight_list:
        :param weight_decay:
        :return:
        '''
        reg_loss = 0

        ir_0 = ir
        vi_0 = vi

        #ir = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(ir)
        #vi = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(vi)

        batch_size = ir.size()[0]
        count = self._tensor_size(ir)
        #count = 1.0

        l_reg1 = (torch.norm(torch.flatten(ir_0), p=1) + torch.norm(torch.flatten(vi_0), p=1)) / batch_size / count
        # l_reg1 = (torch.norm(self.smooth(ir_0), p=1) + torch.norm(self.smooth(vi_0), p=1)) / batch_size / count

        dep = self.r(ir, vi)  # (n,h,w)
        dep_count = torch.ones_like(dep) - dep

        C1 = 0.001
        k = torch.ones_like(dep) / (torch.pow(dep, 2) + C1)
        #k = torch.ones_like(dep) / (torch.abs(dep) + C1)
        #print(k)
        #k=1
        # print(k)

        mir,mvi = self.measure(ir,vi)
        # w1 = 1/(1+torch.exp(k*torch.log(torch.norm(ir_0,p=1,dim=1)+C1)-k*torch.log((torch.norm(vi_0,p=1,dim=1)+C1))))
        #w1 = torch.sigmoid(
        #    k * torch.log(mvi + C1) - k * torch.log(mir+ C1))

        w1 = torch.sigmoid(
            k * mvi - k*mir)
        w2 = 1 - w1
        l_reg2 = torch.norm( (w1 * self.smooth(ir_0) + w2 * self.smooth(vi_0)),
                            p=1) / batch_size / count

        cat = torch.cat([ir_0, vi_0], dim=1)  # (n,c*2,h,w)
        l_reg3 = torch.norm(torch.abs(dep) * self.smooth(cat), p=1) / batch_size / count

        l_reg4 = self.tv(ir) + self.tv(vi)

        reg_loss = 0*l_reg1 + l_reg2*10 + l_reg3*0  + l_reg4*0

        return reg_loss






    def compute_neighnour_differences(self,x):



        w = torch.FloatTensor([
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, -1]]
        ]).unsqueeze(1) #8,1,3,3
        w = w.cuda()

        x = torch.norm(torch.abs(x),p=1,dim=1).unsqueeze(1)

        y = torch.nn.functional.conv2d(x, w, padding=1)
        y = y.abs()
        y = y.sum(dim=1, keepdim=True)

        return x,y


    def measure(self,x1,x2):

        z1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x1)
        z2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x2)

        abs1 = torch.norm(torch.abs(z1), p=1, dim=1).unsqueeze(1)
        abs2 = torch.norm(torch.abs(z2), p=1, dim=1).unsqueeze(1)

        _,y1 = self.compute_neighnour_differences(x1)
        _,y2 = self.compute_neighnour_differences(x2)

        C1 = 0.001


        m1 = (torch.log(y1+1)+1) #* (torch.log(y1+1)+1)



        m2 = (torch.log(y2+1)+1) #* (torch.log(y2+1)+1)

        return m1,m2


    def contrast(self,x1,x2):
        abs1 = torch.mean(torch.abs(x1),dim=1).unsqueeze(1)
        abs2 = torch.mean(torch.abs(x2),dim=1).unsqueeze(1)



        pad =  nn.ReflectionPad2d(2)

        abs1 = pad(abs1)
        abs2 = pad(abs2)

        size = 5
        stride = 1

        window=torch.ones((1,1,size,size))/(size*size)
        window=window.to(x1.device)


        mean_IA=F.conv2d(abs1,window,stride=stride)
        mean_IB=F.conv2d(abs2,window,stride=stride)

        mean_IA_2=F.conv2d(torch.pow(abs1,2),window,stride=stride)
        mean_IB_2=F.conv2d(torch.pow(abs2,2),window,stride=stride)

        var_IA = mean_IA_2 - torch.pow(mean_IA, 2)
        var_IB = mean_IB_2 - torch.pow(mean_IB, 2)



        return var_IA,var_IB










