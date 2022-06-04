import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.spectral_normalization import SpectralNorm
from . import modules as m
import math
from . import pac2 as pac
#from pac import PacConv2d



###############################################################################
# Helper functions
###############################################################################


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer



def define_D(input_nc, ndf, netD, norm='batch', nl='lrelu', init_type='xavier', init_gain=0.02, num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == 'basic_128':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer, nl_layer=nl_layer)
    elif netD == 'basic_256':
        net = D_NLayers(input_nc, ndf, n_layers=3, norm_layer=norm_layer, nl_layer=nl_layer)
    elif netD == 'basic_128_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=2, norm_layer=norm_layer, num_D=num_Ds)
    elif netD == 'basic_256_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer, num_D=num_Ds)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)








class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2**i)))
                layers = self.get_layers(input_nc, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        return sequence

    def forward(self, input, guide=None):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class D_NLayers(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(D_NLayers, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


'''
pix2pix HD
'''




# Defines the PatchGAN discriminator with the specified arguments.


##############################################################################
# Classes
##############################################################################
class RecLoss(nn.Module):
    def __init__(self, use_L2=True):
        super(RecLoss, self).__init__()
        self.use_L2 = use_L2

    def __call__(self, input, target, batch_mean=True):
        if self.use_L2:
            diff = (input - target) ** 2
        else:
            diff = torch.abs(input - target)
        if batch_mean:
            return torch.mean(diff)
        else:
            return torch.mean(torch.mean(torch.mean(diff, dim=1), dim=2), dim=3)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss, all_losses


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck


class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic'):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        max_nchn = 8
        # construct unet structure
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        return self.model(x_with_z)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif upsample == 'nearest':
        upconv = [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck






class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output

###TV_loss#
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
# @torchsnooper.snoop()

import torch
import torch.nn.functional as F
from math import exp
import numpy as np

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # 协方差期望公式：sigma_x=E（X^2）-（EX）^2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # 协方差期望公式：sigma_xy=E(XY）-（EX）*（EY）
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])

    return gauss/gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    #mm是计算两个矩阵/向量的乘积，（m，n）*(n,p)=(m,p)
    #bmm多了一维b，即可以进行批矩阵/向量的乘积
    #matual可以进行高维张量乘法，但是非矩阵的维度会被广播（前提是能广播的情况下）
    #i.e.（j,1,n,m）*(k,m,p)=(j,k,n,p)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()#通道扩展
    return window



# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 3 channel for SSIM
        self.channel = channel
        self.window = create_window(window_size, channel)

    # @torchsnooper.snoop()
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.to(img1.device)
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# Define perceptual loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        #self.slice2 = torch.nn.Sequential()
        #self.slice3 = torch.nn.Sequential()
        #self.slice4 = torch.nn.Sequential()
        #self.slice5 = torch.nn.Sequential()
        for x in range(11):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        '''
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        '''
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        #h_relu2 = self.slice2(h_relu1)
        #h_relu3 = self.slice3(h_relu2)
        #h_relu4 = self.slice4(h_relu3)
        #h_relu5 = self.slice5(h_relu4)
        #out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return [h_relu1]

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        #self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()
        #self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
#################################################模型修改###############################################
###############V0####################




################V2###################


class Encoder(nn.Module):
    def __init__(self, input_nc, outer_nc, ngf,
                 norm_layer=None, nl_layer=None, use_dropout = False, upsample='basic', padding_type='reflect'):
        super().__init__()
        p = 0
        downconv = []

        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
            pad = [nn.ReflectionPad2d(1)]
            pad2 = [nn.ReflectionPad2d(2)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
            pad = [nn.ReplicationPad2d(1)]
            pad2 = [nn.ReplicationPad2d(2)]
        elif padding_type == 'zero':
            p = 1
            pad = None
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, 64,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nl_layer()

        input_block = pad + [nn.Conv2d(input_nc,64,3,1,p)]
        #input_block = [nn.Conv2d(input_nc, 64, 1, 1, p)]

        self.inputblock = nn.Sequential(*input_block)

        #self.down1 = nn.Sequential(*downconv)


        down2 = [downrelu] + pad + [nn.Conv2d(64, 128,
                                              kernel_size=4, stride=2, padding=p)] + [norm_layer(128)]

        self.down2 = nn.Sequential(*down2)

        down3 = [downrelu] + pad + [nn.Conv2d(128, 256,
                                              kernel_size=4, stride=2, padding=p)] + [norm_layer(256)]

        self.down3 = nn.Sequential(*down3)

        down4 = [downrelu] + pad + [nn.Conv2d(256, 512,
                                              kernel_size=4, stride=2, padding=p)] + [norm_layer(512)]

        self.down4 = nn.Sequential(*down4)



        down5 = [downrelu] + pad + [nn.Conv2d(512, 512,
                                              kernel_size=4, stride=2, padding=p)] + [norm_layer(512)]

        self.down5 = nn.Sequential(*down5)


        down6 = [downrelu] + pad + [nn.Conv2d(512, 512,
                                              kernel_size=4, stride=2, padding=p)] + [norm_layer(512)]

        self.down6 = nn.Sequential(*down6)

        down7 = [downrelu] + pad + [nn.Conv2d(512, 512,
                                              kernel_size=4, stride=2, padding=p)]
        self.down7 = nn.Sequential(*down7)


    def forward(self, x):

        x1 = self.inputblock(x)

        x2 = self.down2(x1)

        x3 = self.down3(x2)

        x4 = self.down4(x3)

        #latent = self.down5(x4)

        x5 = self.down5(x4)

        x6 = self.down6(x5)

        latent = self.down7(x6)

        cat_dict = {"cat_1": x1, "cat_2": x2, "cat_3": x3, "cat_4": x4, "cat_5": x5, "cat_6": x6}
        #cat_dict = {"cat_1": x1, "cat_2": x2, "cat_3": x3, "cat_4": x4}

        return cat_dict, latent


class Decoder(nn.Module):
    def __init__(self, input_nc, outer_nc, ngf,
                 norm_layer=None, nl_layer=None, use_dropout = False,upsample='basic', padding_type='reflect'):
        super().__init__()

        uprelu = nl_layer()

        #upconv = upsampleLayer(
        #    64 * 2, outer_nc, upsample=upsample, padding_type=padding_type)
        pad = [nn.ReflectionPad2d(1)]
        #upconv = pad + [nn.Conv2d(64*2, outer_nc, 3, 1, 0)]
        upconv =  [nn.Conv2d(64 * 2, outer_nc, 1, 1, 0)]

        up1 = [uprelu] + upconv + [nn.Tanh()]
        self.up1 = nn.Sequential(*up1)

        up2 = [uprelu] + upsampleLayer(
            128*2 , 64, upsample=upsample, padding_type=padding_type) + [norm_layer(64)]

        if use_dropout:
            up2 += [nn.Dropout(0.5)]

        self.up2 = nn.Sequential(*up2)

        up3 = [uprelu] + upsampleLayer(
            256*2, 128, upsample=upsample, padding_type=padding_type) + [norm_layer(128)]
        if use_dropout:
            up3 += [nn.Dropout(0.5)]
        self.up3 = nn.Sequential(*up3)

        up4 = [uprelu] + upsampleLayer(
            512*2 , 256, upsample=upsample, padding_type=padding_type) + [norm_layer(256)]
        if use_dropout:
            up4 += [nn.Dropout(0.5)]
        self.up4 = nn.Sequential(*up4)

        up5 = [uprelu] + upsampleLayer(
            512*2 , 512, upsample=upsample, padding_type=padding_type) + [norm_layer(512)]
        self.up5 = nn.Sequential(*up5)

        up6 = [uprelu] + upsampleLayer(
            512*2 , 512, upsample=upsample, padding_type=padding_type) + [norm_layer(512)]
        self.up6 = nn.Sequential(*up6)

        up7 = [uprelu] + upsampleLayer(
            512, 512, upsample=upsample, padding_type=padding_type) +  [norm_layer(512)]
        self.up7 = nn.Sequential(*up7)



    def forward(self, x, cat_dict):
        #x = self.up5(x)
        x = self.up7(x)

        x = self.up6(torch.cat([x, cat_dict["cat_6"]], 1))

        x = self.up5(torch.cat([x, cat_dict["cat_5"]], 1))

        #x = self.up4(torch.cat([x, cat_dict["cat_4"]], 1))


        #x = self.up6(x+ cat_dict["cat_6"])

        #x = self.up5(x+ cat_dict["cat_5"])

        x = self.up4(torch.cat([x, cat_dict["cat_4"]], 1))
        # = self.up4(x + cat_dict["cat_4"])
        x = self.up3(torch.cat([x, cat_dict["cat_3"]], 1))
        #x = self.up3(x + cat_dict["cat_3"])
        x = self.up2(torch.cat([x, cat_dict["cat_2"]], 1))
        #x = self.up2(x + cat_dict["cat_2"])
        x = self.up1(torch.cat([x, cat_dict["cat_1"]], 1))
        #x = self.up1(x + cat_dict["cat_1"])

        return x





def define_En(input_nc, output_nc, ngf, netG='unet_128', norm='batch', nl='relu',
              use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='basic',
              padding_type='reflect',n_blocks = 2):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    # if nz == 0:
    #    where_add = 'input'
    if netG == 'v1':
        net = ResEncoder(input_nc,nl =nl_layer,norm = norm_layer, filters=[64, 64, 64, 512])

    if netG == 'v2':
        net = Disentangle(input_nc,nl =nl_layer, norm = norm_layer ,n_blocks=n_blocks)

    if netG == 'v3':
        net = Disentangle3(input_nc,norm_layer = norm_layer)

    if netG == 'v4':
        net = Swap_NET(input_nc,norm_layer = norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)  # 返回经过初始化的网络



class PA_ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(PA_ResnetBlock, self).__init__()


    #def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        #conv_block = []
        p = 0
        if padding_type == 'reflect':
            self.pad = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.pad = nn.ReplicationPad2d(1)
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block1 = pac.PacConv2d(dim, dim, kernel_size=3, padding=p)
        self.norm1 = norm_layer(dim)
        self.act1 = activation

        #conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
        #               norm_layer(dim),
        #               activation]
        if use_dropout:
            self.drop = nn.Dropout(0.5)


        self.conv_block2 = pac.PacConv2d(dim, dim, kernel_size=3, padding=p)
        self.norm2 = norm_layer(dim)

        #conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
        #               norm_layer(dim)]

        #return nn.Sequential(*conv_block)

    def forward(self, x,guide):

        t = x

        guide = self.pad(guide)

        x = self.pad(x)
        x = self.conv_block1(x,guide)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.pad(x)
        x = self.conv_block2(x,guide)
        x = self.norm2(x)


        out = t + x
        return out





class Swap_NET(nn.Module):
    def __init__(self, input_nc ,norm_layer ,n_blocks=2,ngf=64):
        super().__init__()

        n_downsampling = 0
        activation = nn.ReLU(True)
        padding_type = 'reflect'

        #model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.model = nn.Conv2d(input_nc, ngf, kernel_size=1, padding=0)
        ### downsample

        mult = 2 ** n_downsampling
        self.conv1 = PA_ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.act1 = activation

        self.conv2 = PA_ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.act2 = activation

        self.conv3 = PA_ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.act3 = activation



        #后续多加一个卷积层处理
        model2 = [nn.ReflectionPad2d(1),nn.Conv2d(ngf, ngf, kernel_size=3, padding=0),activation]
        #model2 = [m.CBAM(ngf), activation]
        self.model2 = nn.Sequential(*model2)




    def forward(self, x, type, grad):

        if type == "ir":
            #t = torch.pow(x, 2)

            #max = torch.max(t)
            #min = torch.min(t)

            #t = (t - max) / (max - min)

            #guide = x

            t_g = grad

            max = torch.max(t_g)
            min = torch.min(t_g)

            y = (t_g - min) / (max - min)
            # norm_grad = (grad - max_grad) / (max_grad - min_grad)

            guide = x#grad #+ x

        if type == "vi":
            # max_grad = torch.max(grad)

            #min_grad = torch.min(grad)

            #t = grad - min_grad

            t_g = grad

            max = torch.max(t_g)
            min = torch.min(t_g)

            y = (t_g - min) / (max - min)
            # norm_grad = (grad - max_grad) / (max_grad - min_grad)

            guide = grad #+ x

        x = self.model(x)

        x=self.conv1(x,guide)
        x =self.act1(x)

        x = self.conv2(x,guide)
        x = self.act2(x)

        x = self.conv3(x,guide)
        x = self.act3(x)

        x = self.model2(x)


        return x





class Disentangle3(nn.Module):
    def __init__(self, input_nc ,norm_layer ,n_blocks=3,ngf=64):
        super().__init__()

        n_downsampling = 0
        activation = nn.ReLU(True)
        padding_type = 'reflect'

        #model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model = [nn.Conv2d(input_nc, ngf, kernel_size=1, padding=0)]
        ### downsample
        #for i in range(n_downsampling):
        #    mult = 2 ** i
        #    model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
        #              norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer),activation]

        #后续多加一个卷积层处理
        model += [nn.ReflectionPad2d(1),nn.Conv2d(ngf, ngf, kernel_size=3, padding=0),activation]
        #model += [ m.CBAM(ngf), activation]
        self.model = nn.Sequential(*model)




    def forward(self, x):

        x = self.model(x)

        #cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}


        return x


class De3(nn.Module):
    def __init__(self, input_nc ,norm_layer ,n_blocks=2,ngf=64):
        super().__init__()


        activation = nn.ReLU(True)
        n_downsampling = 0
        padding_type = 'reflect'

        model = []

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer),activation]
        self.model = nn.Sequential(*model)

        model2 = []

        #model2 += [nn.Conv2d(ngf, 1, kernel_size=1, padding=0), nn.Sigmoid()]
        model2 += [nn.ReflectionPad2d(1),nn.Conv2d(ngf, 1, kernel_size=3, padding=0), nn.Sigmoid()]
        self.model2 = nn.Sequential(*model2)


    def forward(self, x,y):


        x = self.model(x)

        x_out = self.model2(x)

        y = self.model(y)
        y_out = self.model2(y)


        z_out = self.model2(x+y)


        return z_out,x_out,y_out,x,y



class De4(nn.Module):
    def __init__(self, input_nc ,norm_layer ,n_blocks=3,ngf=64):
        super().__init__()


        activation = nn.LeakyReLU(True)
        n_downsampling = 0
        padding_type = 'reflect'

        model = []

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer),activation]
        self.model = nn.Sequential(*model)

        model2 = []
        ### upsample
        #for i in range(n_downsampling):
        #    mult = 2 ** (n_downsampling - i)
        #    model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
        #                                 output_padding=1),
        #              norm_layer(int(ngf * mult / 2)), activation]
        model2 += [nn.Conv2d(ngf, 1, kernel_size=1, padding=0), nn.Sigmoid()]
        self.model2 = nn.Sequential(*model2)


    def forward(self, x):


        z = self.model(x)

        #x_out = self.model2(x)

        #y = self.model(y)
        #y_out = self.model2(y)


        z_out = self.model2(z)


        return z_out




def define_De(input_nc, output_nc, ngf, netG='unet_128', norm='batch', nl='relu',
              use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='basic',
              padding_type='reflect'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    # if nz == 0:
    #    where_add = 'input'
    if netG == 'v1':
        net = ResDecoder(output_nc,nl =nl_layer,norm = norm_layer, filters=[64, 64, 64, 512])
    if netG == 'v2':
        net1 = Generator_p1(nl=nl_layer)
        net2 = Generator_p2()
        return init_net(net1, init_type, init_gain, gpu_ids), init_net(net2, init_type, init_gain, gpu_ids)

    if netG == 'v3':
        net = De3(input_nc ,norm_layer=norm_layer)

    if netG == 'v4':
        net = De4(input_nc ,norm_layer=norm_layer)


    return init_net(net, init_type, init_gain, gpu_ids)  # 返回经过初始化的网络




class ResEncoder(nn.Module):
    def __init__(self, channel,nl ,norm , filters=[64, 64, 64, 512]):
        super().__init__()


        self.input_layer = nn.Sequential(
            SpectralNorm(nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)),
            norm(filters[0]),
            nl(),
            SpectralNorm(nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)),
        )

        self.input_skip = nn.Sequential(
            SpectralNorm(nn.Conv2d(channel, filters[0], kernel_size=3, padding=1))
        )

        self.residual_conv_1 = m.ResidualConv_pad2(nl,norm,filters[0], filters[1], 1, 1)
        self.residual_conv_2 = m.ResidualConv_pad2(nl,norm,filters[1], filters[2], 1, 1)


    def forward(self, x):


        x = self.input_layer(x) + self.input_skip(x)



        x = self.residual_conv_1(x)



        x = self.residual_conv_2(x)

        #cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}




        return x


class ResDecoder(nn.Module):
    def __init__(self, channel,nl ,norm, filters=[64, 64, 64, 512]):
        super().__init__()





        self.residual_conv1 = m.ResidualConv_pad2(nl,norm,filters[2], filters[1], 1, 1)

        self.residual_conv2 = m.ResidualConv_pad2(nl,norm,filters[1] , filters[0], 1, 1)

        self.residual_conv3 = m.ResidualConv_pad2(nl,norm, filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):

        # Bridge
        x = self.residual_conv1(x)

        x = self.residual_conv2(x)

        x = self.residual_conv3(x)

        x = self.output_layer(x)
        # Decode

        #cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}




        return x


##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

    # Define a resnet block


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

#调制Conv2D
class Conv2DMod(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel


        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused



    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )


        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(
            input, weight, padding=self.padding, groups=batch
        )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out



class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

class Disentangle(nn.Module):
    def __init__(self, channel,nl ,norm ,n_blocks=2,filters=[32, 64,64, 128 ,64]):
        super().__init__()


        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=1, padding=0),
            #nl(),
        )
        res = []

        for i in range(n_blocks):

            in_dim = filters[i]
            out_dim = filters[i+1]
            res += [nn.Conv2d(in_dim,out_dim,3,1,1),nl()]

        self.downsample = nn.Sequential(*res)


        self.structure = nn.Sequential(
            #nl(),
            nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1),
            #norm(filters[3]),
            nl(),
            nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1),
            #norm(8),
            nl(),
        )

        self.texture =nn.Sequential(
            #nl(),
            nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1),
            #norm(filters[3]),
            nl(),
            nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1),
            nl(),
        )





    def forward(self, x):

        x = self.input_layer(x)

        x = self.downsample(x)

        c = self.structure(x)

        t = self.texture(x)


        #cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}


        return c,t




class Generator_p1(nn.Module):
    def __init__(self,nl ,norm=None):
        super().__init__()


        self.conv1  = nn.Sequential(nn.Conv2d(64,64,3,1,1),nl())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32,3,1,1),nl())



    def forward(self, x):


        x = self.conv1(x)



        x = self.conv2(x)





        #cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}

        return x

class Generator_p2(nn.Module):
    def __init__(self):
        super().__init__()




        self.output =  nn.Sequential(
            nn.Conv2d(32, 1, 3,1, 1),
            nn.Tanh(),
        )


    def forward(self, x):



        x = self.output(x)




        #cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}

        return x


class Disentangle2(nn.Module):
    def __init__(self, channel,nl ,norm ,n_blocks=2,filters=[32, 64,64, 64 ,64]):
        super().__init__()


        self.shared_layer1 = nn.Conv2d(1, filters[0], kernel_size=3, padding=1)

        res = []

        for i in range(n_blocks):

            in_dim = filters[i]
            out_dim = filters[i+1]
            res += [nn.Conv2d(in_dim,out_dim,3,1,1),nl()]

        self.shared_layer2 = nn.Sequential(*res)


        #self.shared_layer2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)



        self.structure = nn.Sequential(
            #nl(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
            #norm(filters[3]),
            nl(),
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            #norm(8),
            nl(),
        )

        self.texture1 =nn.Sequential(
            #nl(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
            #norm(filters[3]),
            nl(),
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            nl(),
        )

        self.texture2 =nn.Sequential(
            #nl(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
            #norm(filters[3]),
            nl(),
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            nl(),
        )




    def forward(self, x1,x2):

        x1 = self.shared_layer2(self.shared_layer1(x1))
        x2 = self.shared_layer2(self.shared_layer1(x2))

        s1 = self.structure(x1)
        s2 = self.structure(x2)

        t1 = self.texture1(x1)
        t2 = self.texture2(x2)


        #cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}


        return s1,s2,t1,t2


class Generator2_p1(nn.Module):
    def __init__(self, nl, norm=None):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nl(),nn.Conv2d(64, 32, 3, 1, 1), nl())



    def forward(self, x):
        x = self.conv(x)



        # cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}

        return x


class Generator2_p2(nn.Module):
    def __init__(self):
        super().__init__()

        self.output = nn.Conv2d(32, 1, 3, 1, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x =self.act(self.output(x))


        # cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}

        return x


'''
class Generator(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv1  =GenerateConv(8, 128, 2048,upsample = False)
        self.conv2 = GenerateConv(128, 256, 2048, upsample=False)
        self.conv3 = GenerateConv(256, 384, 2048, upsample=False)
        self.conv4 = GenerateConv(384, 512, 2048, upsample=False)
        self.conv5 = GenerateConv(512, 512, 2048, upsample=True)
        self.conv6 = GenerateConv(512, 512, 2048, upsample=True)
        self.conv7 = GenerateConv(512, 256, 2048, upsample=True)
        self.conv8 = GenerateConv(256, 128, 2048, upsample=True)

        self.output =  nn.Sequential(
            nn.Conv2d(128, 1, 1, 1),
            nn.Tanh(),
        )


    def forward(self, x,s):


        x = self.conv1(x,s)



        x = self.conv2(x,s)



        x = self.conv3(x,s)
        x = self.conv4(x,s)
        x = self.conv5(x,s)
        x = self.conv6(x,s)
        x = self.conv7(x,s)


        x = self.conv8(x,s)


        x = self.output(x)




        #cat_dict = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}

        return x
'''

class Discriminator_fgan(nn.Module):
    def __init__(self, input_nc,size = 9216):
        super().__init__()

        #self.block1 = nn.Sequential(*[nn.Conv2d(input_nc, 32, 3, 2, 0),nl_layer()])

        self.block1 = SpectralNorm(nn.Conv2d(input_nc, 32, 3, 2, 0))
        self.bn_relu1 = nn.LeakyReLU(0.2, inplace=True)

        #self.block2 = nn.Sequential(*[nn.Conv2d(32, 64, 3, 2, 0), nl_layer(),norm_layer(64)])

        self.block2 = SpectralNorm(nn.Conv2d(32, 64, 3, 2, 0))
        self.bn_relu2 = nn.Sequential(*[nn.LeakyReLU(0.2, inplace=True),nn.BatchNorm2d(64)])

        #self.block3 = nn.Sequential(*[nn.Conv2d(64, 128, 3, 2, 0), nl_layer(), norm_layer(128)])

        self.block3 = SpectralNorm(nn.Conv2d(64, 128, 3, 2, 0))
        self.bn_relu3 = nn.Sequential(*[nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(128)])

        #self.block4 = nn.Sequential(*[nn.Conv2d(128, 256, 3, 2, 0), nl_layer(), norm_layer(256)])

        self.block4 = SpectralNorm(nn.Conv2d(128, 256, 3, 2, 0))
        self.bn_relu4 = nn.Sequential(*[nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(256)])

        # The height and width of downsampled image

        self.adv_layer = EqualLinear(size, 1, bias_init=1)
        self.down = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        # construct unet structure


    def forward(self, x):
        out = self.block1(x)
        out = self.bn_relu1(out)
        out = self.block2(out)
        out = self.bn_relu2(out)
        out = self.block3(out)
        out = self.bn_relu3(out)
        out = self.block4(out)
        out = self.bn_relu4(out)

        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class Discriminator_fgan2(nn.Module):
    def __init__(self, input_nc,size = 9216):
        super().__init__()

        #self.block1 = nn.Sequential(*[nn.Conv2d(input_nc, 32, 3, 2, 0),nl_layer()])

        self.down = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

        self.block1 = nn.Conv2d(input_nc, 32, 3, 2, 0) #120
        self.bn_relu1 = nn.LeakyReLU(0.2, inplace=True)

        #self.block2 = nn.Sequential(*[nn.Conv2d(32, 64, 3, 2, 0), nl_layer(),norm_layer(64)])

        self.block2 = pac.PacConv2d(32, 64, 3, 2, 0) #60
        self.bn_relu2 = nn.Sequential(*[nn.LeakyReLU(0.2, inplace=True),nn.BatchNorm2d(64)])

        #self.block3 = nn.Sequential(*[nn.Conv2d(64, 128, 3, 2, 0), nl_layer(), norm_layer(128)])

        self.block3 = pac.PacConv2d(64, 128, 3, 2, 0)#30
        self.bn_relu3 = nn.Sequential(*[nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(128)])

        #self.block4 = nn.Sequential(*[nn.Conv2d(128, 256, 3, 2, 0), nl_layer(), norm_layer(256)])

        self.block4 = pac.PacConv2d(128, 256, 3, 2, 0)#15
        self.bn_relu4 = nn.Sequential(*[nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(256)])

        # The height and width of downsampled image

        self.adv_layer = EqualLinear(size, 1, bias_init=1)


        # construct unet structure


    def forward(self, x, guide):


        out = self.block1(x)
        out = self.bn_relu1(out)



        guide = self.down(guide)

        out = self.block2(out,guide)
        out = self.bn_relu2(out)



        guide = self.down(guide)
        out = self.block3(out,guide)
        out = self.bn_relu3(out)

        guide = self.down(guide)
        out = self.block4(out,guide)
        out = self.bn_relu4(out)




        out = out.view(out.shape[0], -1)


        validity = self.adv_layer(out)
        return validity


def define_D2(input_nc, ndf, netD, norm='batch', nl='lrelu', init_type='xavier', init_gain=0.02, num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)


    net =  Discriminator_fgan2(1,size=9216)

    return init_net(net, init_type, init_gain, gpu_ids)
#####################隐空间编码融合########################

class fusion_latent(nn.Module):
    def __init__(self,input_nc,output_nc):
        super().__init__()

        self.fuse = nn.Conv2d(input_nc,output_nc,1,1,0)


    def forward(self, x1,x2):
        x = torch.cat([x1,x2],1)
        x = self.fuse(x)

        return x

def define_fuselat(input_nc,output_nc, netG='unet_128', init_type='xavier', init_gain=0.02, gpu_ids=[]):
    net = None

    # if nz == 0:
    #    where_add = 'input'

    net = fusion_latent(input_nc,output_nc)

    return init_net(net, init_type, init_gain, gpu_ids)  # 返回经过初始化的网络



class ResidualConv(nn.Module):
    def __init__(self, norm,nl,input_dim, output_dim, stride, padding):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=0
            ),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=0),
        )
        self.conv_skip = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=0),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return (self.conv_block(x) + self.conv_skip(x))* (1 / math.sqrt(2))







def define_SimpleEn(input_nc, output_nc, ngf, netG='unet_128', norm='batch', nl='relu',
              use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='basic',
              padding_type='reflect',n_blocks = 2):

    net = SimpleEncoder()

    if netG=='C':
        net = SimpleCommon()

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    return net  # 返回经过初始化的网络


def define_SimpleDe(input_nc, output_nc, ngf, netG='unet_128', norm='batch', nl='relu',
              use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='basic',
              padding_type='reflect',n_blocks = 2):

    net = SimpleDecoder()

    if netG=='v2':
        net = SimpleDecoder2()



    return init_net(net, init_type, init_gain, gpu_ids)


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        norm = 'instance'
        norm_layer = get_norm_layer(norm_type=norm)
        nl = 'relu'  # use leaky relu for D
        nl_layer = get_non_linearity(layer_type=nl)

        pad = [nn.ReflectionPad2d(1)]

        for p in models.vgg19(pretrained=True).parameters():
            p.requires_grad = True

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        conv1 = vgg_pretrained_features[0]
        conv1.padding = (0, 0)

        conv2 = vgg_pretrained_features[2]
        conv2.padding = (0, 0)

        conv3 = vgg_pretrained_features[5]
        conv3.padding = (0, 0)

        slice1 = pad + [conv1] +[norm_layer(64)] + [nl_layer()]
        slice2 = pad + [conv2] + [norm_layer(64)] + [nl_layer()]
        slice3 = pad + [conv3] + [norm_layer(128)] + [nl_layer()]

        self.slice1 = nn.Sequential(*slice1)
        self.slice2 = nn.Sequential(*slice2)
        self.slice3 = nn.Sequential(*slice3)


    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        x = self.slice1(x)
        x = self.slice2(x)
        x = self.slice3(x)

        return x


class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        norm = 'instance'
        norm_layer = get_norm_layer(norm_type=norm)
        nl = 'relu'  # use leaky relu for D
        nl_layer = get_non_linearity(layer_type=nl)

        pad = [nn.ReflectionPad2d(1)]

        conv1 = pad+ [nn.Conv2d(128,64,3,1,0)] + [norm_layer(64)] + [nl_layer()]
        self.conv1 = nn.Sequential(*conv1)

        conv2 = pad+ [nn.Conv2d(64, 32, 3, 1, 0)] + [norm_layer(32)] + [nl_layer()]
        self.conv2 = nn.Sequential(*conv2)

        conv3 = pad+ [nn.Conv2d(32, 32, 3, 1, 0)] + [norm_layer(32)] + [nl_layer()]
        self.conv3 = nn.Sequential(*conv3)

        conv4 = [nn.Conv2d(32, 1, 1, 1, 0)] + [nn.Sigmoid()]

        self.conv4 = nn.Sequential(*conv4)

    def forward(self, x,y):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x_out = self.conv4(x)


        y = self.conv1(y)

        y = self.conv2(y)

        y = self.conv3(y)

        y_out = self.conv4(y)

        z_out = self.conv4(x+y)


        return z_out,x_out,y_out,x,y



class SimpleDecoder2(nn.Module):
    def __init__(self):
        super().__init__()
        norm = 'instance'
        norm_layer = get_norm_layer(norm_type=norm)
        nl = 'relu'  # use leaky relu for D
        nl_layer = get_non_linearity(layer_type=nl)

        pad = [nn.ReflectionPad2d(1)]

        conv1 = pad+ [nn.Conv2d(128,64,3,1,0)] + [norm_layer(64)] + [nl_layer()]
        self.conv1 = nn.Sequential(*conv1)

        conv2 = pad+ [nn.Conv2d(64, 32, 3, 1, 0)] + [norm_layer(32)] + [nl_layer()]
        self.conv2 = nn.Sequential(*conv2)

        conv3 = pad+ [nn.Conv2d(32, 32, 3, 1, 0)] + [norm_layer(32)] + [nl_layer()]
        self.conv3 = nn.Sequential(*conv3)

        conv4 = pad+ [nn.Conv2d(32, 1, 3, 1, 0)] + [nn.Sigmoid()]

        self.conv4 = nn.Sequential(*conv4)

    def forward(self, x):



        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        z_out = self.conv4(x)


        return z_out


class SimpleCommon(nn.Module):
    def __init__(self):
        super().__init__()

        norm = 'instance'
        norm_layer = get_norm_layer(norm_type=norm)
        nl = 'relu'  # use leaky relu for D
        nl_layer = get_non_linearity(layer_type=nl)

        pad = [nn.ReflectionPad2d(1)]

        for p in models.vgg19(pretrained=True).parameters():
            p.requires_grad = True

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        conv1 = nn.Conv2d(1,64,3,1,0)
        conv1.padding = (0, 0)

        conv2 = vgg_pretrained_features[2]
        conv2.padding = (0, 0)

        conv3 = vgg_pretrained_features[5]
        conv3.padding = (0, 0)

        slice1 = pad + [conv1] +[norm_layer(64)] + [nl_layer()]
        slice2 = pad + [conv2] + [norm_layer(64)] + [nl_layer()]
        slice3 = pad + [conv3] + [norm_layer(128)] + [nl_layer()]

        self.slice1 = nn.Sequential(*slice1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.slice2 = nn.Sequential(*slice2)
        self.slice3 = nn.Sequential(*slice3)


    def forward(self, x):

        x = self.slice1(x)
        x = self.slice2(x)
        x = self.slice3(x)

        return x
