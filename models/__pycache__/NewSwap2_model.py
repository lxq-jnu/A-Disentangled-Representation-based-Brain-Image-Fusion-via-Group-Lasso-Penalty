import torch
from .base_model import BaseModel
from . import networks_swap as networks
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import random
###这个测试了使用Mask的情况

def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, channel, height, width = img.shape
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, height - c_h)
        c_x = random.randrange(0, width - c_w)

        cropped = img[:, :, c_y: c_y + c_h, c_x: c_x + c_w]
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

    return patches


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

class NewSwapModel(BaseModel):
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
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake','G_VGG','SSIM','grad','z_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A','rec_A','rec_fake_A', 'real_B','rec_B','rec_fake_B','fake_F','rec_ir_sal','rec_vi_sal']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['En','De', 'DIR','DVI']
        else:  # during test time, only load G
            self.model_names = ['En','De']
        # define networks (both generator and discriminator)

        #self.netEn = networks.define_En(opt.input_nc * 1, opt.output_nc, opt.ngf, netG='v2',
        #                                   norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout,
        #                                   init_type=opt.init_type, init_gain=opt.init_gain,
        #                                   gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)

        net = networks.Disentangle2().to(self.gpu_ids[0])
        self.netEn = torch.nn.DataParallel(net,self.gpu_ids)



        self.netDe = networks.define_De(64, opt.output_nc, opt.ngf, netG='v2',
                                           norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout,
                                           init_type=opt.init_type, init_gain=opt.init_gain,
                                           gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)

        self.netD = networks.init_net(networks.SA_Discriminator(),init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        #'''
        self.netDVI = networks.define_D(opt.input_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                      init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)

        self.netDIR = networks.define_D(opt.input_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                      init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds,
                                      gpu_ids=self.gpu_ids)
        #'''


        if self.isTrain:
            # define loss functions
            #if not opt.no_ganFeat_loss:
            #    self.criterionGAN = networks.GANLoss_hd(gan_mode=opt.gan_mode).to(self.device)
            #else:
            #    self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            #self.criterionGAN = networks.GANLoss_hd(gan_mode=opt.gan_mode).to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionSSIM = networks.SSIM()  # SSIM 结构相似性损失作为图像的重构损失
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionTV = networks.TVLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_En = torch.optim.Adam(self.netEn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


            self.optimizer_De = torch.optim.Adam(self.netDe.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


            self.optimizer_DVI = torch.optim.Adam(self.netDVI.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DIR = torch.optim.Adam(self.netDIR.parameters(), lr=opt.lr*4, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*4, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_En)

            self.optimizers.append(self.optimizer_De)

            self.optimizers.append(self.optimizer_DVI)
            self.optimizers.append(self.optimizer_DIR)
            self.optimizers.append(self.optimizer_D)

    def set_ewc(self,ewc):
        self.ewc = ewc
        self.train_mode = "ewc"

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if 'M' in input.keys():
            self.real_M = input['M'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test2(self,real_A,real_B):
        with torch.no_grad():
            # concat_AB = torch.cat([self.real_A,self.real_B],1)
            self.real_A =real_A
            self.real_B = real_B

            real_A = torch.cat([self.real_A, self.real_A, self.real_A], 1)
            real_B = torch.cat([self.real_B, self.real_B, self.real_B], 1)

            self.ir_back, self.vi_back, self.ir_sal, self.vi_sal = self.netEn(real_A, real_B)

            self.rec_A = self.netDe(self.ir_back+self.ir_sal)
            self.rec_B = self.netDe(self.vi_back+self.vi_sal)

            self.fake_F = self.netDe(self.vi_back + self.ir_sal + self.vi_sal)

            #cat_dict,z = self.encode(concat_AB)
            #self.fake_F = self.netG(concat_AB)

            return self.real_A, self.fake_F, self.real_B
            #return self.ir_back, self.vi_back,self.ir_sal,self.ir_p


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        real_A = torch.cat([self.real_A,self.real_A,self.real_A],1)
        real_B = torch.cat([self.real_B, self.real_B, self.real_B], 1)

        self.ir_back,self.vi_back,self.ir_sal,self.vi_sal = self.netEn(real_A,real_B)

        self.rec_A = self.netDe(self.ir_back+self.ir_sal)
        self.rec_B = self.netDe(self.vi_back+ self.vi_sal)

        self.rec_fake_A = self.netDe(self.vi_back+self.ir_sal)
        self.rec_fake_B = self.netDe(self.ir_back+ self.vi_sal)

        self.rec_ir_sal =self.netDe(self.ir_sal)
        self.rec_vi_sal = self.netDe(self.vi_sal)

       # self.fake_F = self.netDe(torch.cat([self.vi_back, self.ir_sal+self.vi_sal], 1))
        self.fake_F = self.netDe(self.vi_back+self.ir_sal+ self.vi_sal)

        '''
        vi_p = F.interpolate(
            self.vi_p, size=(120, 120), mode="bilinear", align_corners=False
        )
        '''


        #self.rec_fake_A =ir_p*self.fake_F+(1-ir_p)*self.real_A
        #self.rec_fake_B = ir_p * self.fake_F + (1 - ir_p) * self.real_B

        fake_F = torch.cat([self.fake_F, self.fake_F, self.fake_F], 1)

        _,_, self.F_ir_sal, self.F_vi_sal = self.netEn(fake_F,fake_F)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        #fake_AA = torch.cat(self.rec_A)
        fake_AA = self.rec_A
        fake_AF = self.rec_fake_A
        #fake_AF = torch.cat((self.real_A, self.fake_F), 1)
        #fake_BB = torch.cat(self.rec_B)
        fake_BB = self.rec_B
        fake_BF = self.rec_fake_B
        #fake_BF = torch.cat((self.real_B, self.fake_F), 1)

        #fake_AS = torch.cat(self.rec_ir_sal)
        #fake_BS = torch.cat( self.rec_vi_sal)
        fake_AS = self.rec_ir_sal
        fake_BS = self.rec_vi_sal
        #pred_fake1 = self.netD(fake_AA1.detach())
        #pred_fake2 = self.netD(fake_AA2.detach())

        #pred_fake3 = self.netD(fake_BB1.detach())
        #pred_fake4 = self.netD(fake_BB2.detach())


        #pred_fake1 = self.netDIR(fake_AA)
        #pred_fake2 = self.netDIR(fake_AF)
        #pred_fake3 = self.netDVI(fake_BB)
        #pred_fake4 = self.netDVI(fake_BF)
        pred_fake5 = self.netDIR(fake_AS)
        pred_fake6 = self.netDVI(fake_BS)

        #pred_fake2 = self.netDVI(self.rec_B.detach())
        #pred_fake3 = self.netDIR(self.rec_fake_A.detach())

        #self.loss_D_fake1,_  = self.criterionGAN(pred_fake1, False)
        #self.loss_D_fake2, _ = self.criterionGAN(pred_fake2, False)
        #self.loss_D_fake3,_  = self.criterionGAN(pred_fake3, False)
        #self.loss_D_fake4, _ = self.criterionGAN(pred_fake4, False)
        self.loss_D_fake5,_  = self.criterionGAN(pred_fake5, False)
        self.loss_D_fake6, _ = self.criterionGAN(pred_fake6, False)
        #self.loss_D_fake3, _ = self.criterionGAN(pred_fake3, False)



        pred_content = self.netD(self.ir_back.detach())
        self.loss_D_fake_content,_ = self.criterionGAN(pred_content,False)


        self.loss_D_fake =self.loss_D_fake5+self.loss_D_fake6 + self.loss_D_fake_content

        # self.loss_D_fake1+self.loss_D_fake2+self.loss_D_fake3+self.loss_D_fake4\
                           #+self.loss_D_fake5+self.loss_D_fake6 + self.loss_D_fake_content#self.loss_D_fake3

        # Real
        real_AA = self.real_A
        real_BB = self.real_B
        #pred_real1 = self.netDIR(real_AA)
        #pred_real2 = self.netDIR(real_BB)

        pred_real_content = self.netD(self.vi_back)
        pred_real_content2 = self.netD(self.ir_sal)
        pred_real_content3 = self.netD(self.vi_sal)
        self.loss_D_real_content,_ = self.criterionGAN(pred_real_content,True)
        self.loss_D_real_content2, _ = self.criterionGAN(pred_real_content2, True)
        self.loss_D_real_content3, _ = self.criterionGAN(pred_real_content3, True)

        #self.loss_D_real1,_  = self.criterionGAN(pred_real1, True)
        #self.loss_D_real2, _ = self.criterionGAN(pred_real2, True)

        self.loss_D_real = self.loss_D_real_content+self.loss_D_real_content2+self.loss_D_real_content3

        #self.loss_D_real1*2+ self.loss_D_real2*2+ self.loss_D_real_content




        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real)



        self.loss_D.backward(retain_graph = True)




    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        fake_AA = self.rec_A
        fake_AF = self.rec_fake_A
        # fake_AF = torch.cat(self.real_A)
        fake_BB = self.real_B
        fake_BF = self.rec_fake_B
        # fake_BF = torch.cat(self.fake_F)
        fake_AS = self.rec_ir_sal
        fake_BS = self.rec_vi_sal

        #fake_AA = torch.cat(self.rec_A)
        #fake_AF = torch.cat(self.real_A)
        #fake_BB = torch.cat(self.real_B)
        #fake_BF = torch.cat(self.fake_F)
        ##fake_AS = torch.cat( self.rec_ir_sal)
        #fake_BS = torch.cat(self.rec_vi_sal)
        # pred_fake1 = self.netD(fake_AA1.detach())
        # pred_fake2 = self.netD(fake_AA2.detach())

        # pred_fake3 = self.netD(fake_BB1.detach())
        # pred_fake4 = self.netD(fake_BB2.detach())

        #pred_fake1 = self.netDIR(fake_AA)
        #pred_fake2 = self.netDIR(fake_AF)
        #pred_fake3 = self.netDVI(fake_BB)
        #pred_fake4 = self.netDVI(fake_BF)
        pred_fake5 = self.netDIR(fake_AS)
        pred_fake6 = self.netDVI(fake_BS)

        # pred_fake2 = self.netDVI(self.rec_B.detach())
        # pred_fake3 = self.netDIR(self.rec_fake_A.detach())

        #self.loss_G_GAN1, _ = self.criterionGAN(pred_fake1, True)
        #self.loss_G_GAN2, _ = self.criterionGAN(pred_fake2, True)
        #self.loss_G_GAN3, _ = self.criterionGAN(pred_fake3, True)
        #self.loss_G_GAN4, _ = self.criterionGAN(pred_fake4, True)
        self.loss_G_GAN6, _ = self.criterionGAN(pred_fake5, True)
        self.loss_G_GAN7, _ = self.criterionGAN(pred_fake6, True)
        # self.loss_D_fake3, _ = self.criterionGAN(pred_fake3, False)


        #pred_fake3 = self.netDIR(self.rec_fake_A)

        pred_fake_content = self.netD(self.ir_back)
        pred_fake_content2 = self.netD(self.ir_sal)
        pred_fake_content3 = self.netD(self.vi_sal)


        self.loss_G_GAN5, _ = self.criterionGAN(pred_fake_content, True)
        self.loss_G_GAN8, _ = self.criterionGAN(pred_fake_content2, False)
        self.loss_G_GAN9, _ = self.criterionGAN(pred_fake_content3, False)

        self.loss_G_GAN = self.loss_G_GAN6+self.loss_G_GAN7 + self.loss_G_GAN5+self.loss_G_GAN9+self.loss_G_GAN8
        #self.loss_G_GAN1+ self.loss_G_GAN2 + self.loss_G_GAN3 + self.loss_G_GAN4+ self.loss_G_GAN5\

        # Second, G(A) = B

        self.loss_G_L1 = (self.criterionL1(self.rec_A, self.real_A)+ self.criterionL1(self.rec_B, self.real_B)
                          +self.criterionL1(self.rec_fake_A, self.real_A)+self.criterionL1(self.rec_fake_B, self.real_B)
                          )* self.opt.lambda_L1

        self.loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            cat_fake1 = torch.cat([self.rec_A,self.rec_A,self.rec_A], 1)
            cat_real1 = torch.cat([self.real_A, self.real_A, self.real_A], 1)
            cat_fake2 = torch.cat([self.rec_B, self.rec_B, self.rec_B], 1)
            cat_real2 = torch.cat([self.real_B, self.real_B, self.real_B], 1)
            cat_fake3 = torch.cat([self.rec_fake_A,self.rec_fake_A,self.rec_fake_A], 1)
            cat_fake4= torch.cat([self.rec_fake_B, self.rec_fake_B, self.rec_fake_B], 1)


            self.loss_G_VGG = (self.criterionVGG(cat_fake1, cat_real1) + self.criterionVGG(cat_fake2, cat_real2)
                               #+self.criterionVGG(cat_fake3, cat_real1)+self.criterionVGG(cat_fake4, cat_real2)
                               )* self.opt.lambda_feat

        self.loss_z_L1 = 0
        if self.opt.lambda_z > 0.0:
            #self.loss_z_L1 = (self.criterionL1(self.c_ir_fake, self.c_ir)+ self.criterionL1(self.c_vi_fake, self.c_vi)+
            #                  self.criterionL1(self.s_ir_fake, self.s_ir)+ self.criterionL1(self.s_vi_fake, self.s_vi))* self.opt.lambda_z

            #self.loss_z_L1 = self.criterionL2(self.ir_p_up, self.real_M)* self.opt.lambda_z + self.criterionL2(self.F_ir_p,self.ir_p)*10
            self.loss_z_L1 =  self.criterionL2(self.F_vi_sal, torch.max(self.ir_sal,self.vi_sal)) * 10
        self.loss_SSIM = 0
        if self.opt.lambda_SSIM > 0.0:
            self.loss_SSIM = (1 - self.criterionSSIM(self.rec_A, self.real_A)) * self.opt.lambda_SSIM\
                             +(1 - self.criterionSSIM(self.rec_B, self.real_B)) * self.opt.lambda_SSIM


        self.loss_TV = 0
        '''
        if self.opt.lambda_grad > 0.0:
            self.loss_TV = self.criterionTV(self.fake_F - self.real_F)
        '''

        self.loss_grad = 0
        if self.opt.lambda_grad > 0.0:
            self.lap = LaplacianConv()
            self.realA_grad = self.lap(self.real_A)
            self.realB_grad = self.lap(self.real_B)

            self.max_grad = torch.max(self.realA_grad,self.realB_grad)
            self.fakeF_grad = self.lap(self.fake_F)
            self.loss_grad = self.criterionL1(self.fakeF_grad, self.max_grad) * self.opt.lambda_grad\
                            # +self.criterionL1(self.fakeF_grad, self.realB_grad) * self.opt.lambda_grad




        # combine loss and calculate gradients


        if self.train_mode == "ewc":
            #print("Hello")
            self.loss_EWC = (self.ewc.penalty_EN(self.netEn) + self.ewc.penalty_DE(self.netDe))
            self.loss_G = self.loss_G_GAN + self.loss_G_L1+self.loss_G_VGG+self.loss_SSIM + self.loss_grad +self.loss_EWC*1000
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_VGG+self.loss_z_L1
            self.loss_EWC = 0

        with torch.autograd.set_detect_anomaly(True):
            self.loss_G.backward(retain_graph=True)


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)


        self.set_requires_grad(self.netDIR, True)  # enable backprop for D
        self.set_requires_grad(self.netDVI, True)

        self.set_requires_grad(self.netD, True)  # enable backprop for D


        self.optimizer_DIR.zero_grad()  # set D's gradients to zero
        self.optimizer_DVI.zero_grad()  # set D's gradients to zero
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()
        self.optimizer_DIR.step()  # update D's weights
        self.optimizer_DVI.step()
        self.optimizer_D.step()  # update D's weights

        self.set_requires_grad(self.netDIR, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netDVI, False)
        self.set_requires_grad(self.netD, False)
        #self.set_requires_grad(self.netDVI, False)  # D requires no gradients when optimizing G
        self.optimizer_En.zero_grad()  # set G's gradients to zero
        #self.optimizer_EnSIR.zero_grad()  # set G's gradients to zero
        #self.optimizer_EnSVI.zero_grad()  # set G's gradients to zero
        self.optimizer_De.zero_grad()
        self.backward_G()  # calculate graidents for G
        self.optimizer_En.step()  # udpate G's weights
        #self.optimizer_EnSIR.step()
        #self.optimizer_EnSVI.step()
        self.optimizer_De.step()


        self.optimizer_En.zero_grad()  # set G's gradients to zero
        self.optimizer_De.zero_grad()




class LaplacianConv(nn.Module):
    # 仅有一个参数，通道，用于自定义算子模板的通道
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展到3个维度
        kernel = np.repeat(kernel, self.channels, axis=0)  # 3个通道都是同一个算子
        self.weight = nn.Parameter(data=kernel, requires_grad=False)  # 不允许求导更新参数，保持常量

    def __call__(self, x):
        # 第一个参数为输入，由于这里只是测试，随便打开的一张图只有3个维度，pytorch需要处理4个维度的样本，因此需要添加一个样本数量的维度
        # padding2是为了能够照顾到边角的元素
        self.weight.data = self.weight.data.to(x.device)
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        return x


