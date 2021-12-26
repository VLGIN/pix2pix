import torch
from torch import nn
import functools

class Identity(nn.Module):
    def forward(self, x):
        return x


class UnetSkipConnectionBlock(nn.Module):
    """
        Unet Skip connection block
        |-- downsampling --|submodule|-- upsampling--|
    """
    def __init__(self, outer_num_channel, inner_num_channel, input_num_channel=None,
                submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_num_channel is None:
          input_num_channel = outer_num_channel
        # Down sample net
        downconv = nn.Conv2d(input_num_channel, inner_num_channel, kernel_size=4,
                            stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_num_channel)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_num_channel)

        # If is the most outer block
        if outermost:
            upconv = nn.ConvTranspose2d(inner_num_channel * 2, outer_num_channel, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost: # if is the most inner block (the last block with no submodules)
            upconv = nn.ConvTranspose2d(inner_num_channel, outer_num_channel,
                                        kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else: # inner block
            upconv = nn.ConvTranspose2d(inner_num_channel * 2, outer_num_channel, 
                                        kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """
        Unet generator built of multiple Unet skip connection blocks
    """

    def __init__(self, input_num_channel, output_num_channel, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
                use_dropout=False):
        super(UnetGenerator, self).__init__()
                
        #inner most block
        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_num_channel=None, submodule=None, norm_layer=norm_layer, innermost=True)
        #add (num_downs - 5) more blocks above of inner most block
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_num_channel=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        #add 4 more blocks and gradually reduce the number of channel from ngf * 8 -> ngf
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, input_num_channel=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, input_num_channel=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf*2, input_num_channel=None, submodule=unet_block, norm_layer=norm_layer)
        #outermost block
        self.model = UnetSkipConnectionBlock(output_num_channel, ngf, input_num_channel=input_num_channel, submodule=unet_block, outermost=True, norm_layer=norm_layer)
    
    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    """
        Resnet Block
    """
    def __init__(self, dim, padding_style, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_style, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, dim, padding_style, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_style == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_style == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_style == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_style)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_style == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_style == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_style == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_style)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    """
        Resnet Generator
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_style=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    """
        PatchGAN discriminator
    """

    def __init__(self, input_num_channel, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_num_channel, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf  * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
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

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
