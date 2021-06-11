import torchvision
import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
n_latent = 11


channels = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256,
    128: 128,
    256: 64,
    512: 32,
    1024: 16,
}

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


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
        bias = self.bias*self.lr_mul if self.bias is not None else None
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=bias
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.use_style = use_style

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        if use_style:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        else:
            self.modulation = nn.Parameter(torch.Tensor(1, 1, in_channel, 1, 1).fill_(1))

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if self.use_style:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            weight = self.scale * self.weight * style
        else:
            weight = self.scale * self.weight.expand(batch,-1,-1,-1,-1) * self.modulation

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, style_dim):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, style_dim))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, n_latent)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        self.use_style = use_style

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            use_style=use_style,
            upsample=upsample,
            downsample=downsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        #if use_style:
        #    self.noise = NoiseInjection()
        #else:
        #    self.noise = None
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style=None, noise=None):
        out = self.conv(input, style)
        #if self.use_style:
        #    out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class StyledResBlock(nn.Module):
    def __init__(self, in_channel, style_dim, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()

        self.conv1 = StyledConv(in_channel, in_channel, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)
        self.conv2 = StyledConv(in_channel, in_channel, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)

    def forward(self, input, style):
        out = self.conv1(input, style)
        out = self.conv2(out, style)
        out = (out + input) / math.sqrt(2)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        num_down, 
        latent_dim,
        n_mlp,
        n_res,
        channel_multiplier=1,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()
        self.size = size

        style_dim = 512
        
        mapping = [EqualLinear(latent_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu')]
        for i in range(n_mlp-1):
            mapping.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))

        self.mapping = nn.Sequential(*mapping)

        self.encoder = Encoder(size, latent_dim, num_down, n_res, channel_multiplier)

        self.log_size = int(math.log(size, 2)) #7
        in_log_size = self.log_size - num_down #7-2 or 7-3
        in_size = 2 ** in_log_size

        in_channel = channels[in_size]
        self.adain_bottleneck = nn.ModuleList()
        for i in range(n_res):
            self.adain_bottleneck.append(StyledResBlock(in_channel, style_dim))

        self.conv1 = StyledConv(in_channel, in_channel, 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(in_channel, style_dim, upsample=False)

        self.num_layers = (self.log_size - in_log_size) * 2 + 1 #7

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        #self.noises = nn.Module()


        #for layer_idx in range(self.num_layers):
        #    res = (layer_idx + (in_log_size*2+1)) // 2 #2,3,3,5 ... -> 4,5,5,6 ...
        #    shape = [1, 1, 2 ** res, 2 ** res]
        #    self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(in_log_size+1, self.log_size + 1):
            out_channel = channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

    def style_encode(self, input):
        return self.encoder(input)[1]

    def encode(self, input):
        return self.encoder(input)

    def forward(self, input, z=None):
        content, style = self.encode(input)
        if z is None:
            out = self.decode(content, style)
        else:
            out = self.decode(content, z)

        return out, content, style

    def decode(self, input, styles, use_mapping=True):
        if use_mapping:
            styles = self.mapping(styles)
        #styles = styles.repeat(1, n_latent).view(styles.size(0), n_latent, -1)
        out = input
        i = 0
        for conv in self.adain_bottleneck:
            out = conv(out, styles)
            i += 1

        out = self.conv1(out, styles, noise=None)
        skip = self.to_rgb1(out, styles)
        i += 2

        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, styles, noise=None)
            out = conv2(out, styles, noise=None)
            skip = to_rgb(out, styles, skip)

            i += 3

        image = skip
        return image

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class InResBlock(nn.Module):
    def __init__(self, in_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = StyledConv(in_channel, in_channel, 3, None, blur_kernel=blur_kernel, demodulate=True, use_style=False)
        self.conv2 = StyledConv(in_channel, in_channel, 3, None, blur_kernel=blur_kernel, demodulate=True, use_style=False)

    def forward(self, input):
        out = self.conv1(input, None)
        out = self.conv2(out, None)
        out = (out + input) / math.sqrt(2)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
            )
        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is None:
            skip = input
        else:
            skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.size = size
        l_branch = self.make_net_(32)
        l_branch += [ConvLayer(channels[32], 1, 1, activate=False)]
        self.l_branch = nn.Sequential(*l_branch)


        g_branch = self.make_net_(8)
        self.g_branch = nn.Sequential(*g_branch)
        self.g_adv = ConvLayer(channels[8], 1, 1, activate=False)

        self.g_std = nn.Sequential(ConvLayer(channels[8], channels[4], 3, downsample=True),
                      nn.Flatten(),
                      EqualLinear(channels[4] * 4 * 4, 128, activation='fused_lrelu'), 
                      )
        self.g_final = EqualLinear(128, 1, activation=False)


    def make_net_(self, out_size):
        size = self.size
        convs = [ConvLayer(3, channels[size], 1)]
        log_size = int(math.log(size, 2))
        out_log_size = int(math.log(out_size, 2))
        in_channel = channels[size]

        for i in range(log_size, out_log_size, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        return convs

    def forward(self, x):
        l_adv = self.l_branch(x)

        g_act = self.g_branch(x)
        g_adv = self.g_adv(g_act)

        output = self.g_std(g_act)
        g_stddev = torch.sqrt(output.var(0, keepdim=True, unbiased=False) + 1e-8).repeat(x.size(0),1)
        g_std = self.g_final(g_stddev)
        return [l_adv, g_adv, g_std]



class Encoder(nn.Module):
    def __init__(self, size, latent_dim, num_down, n_res, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        stem = [ConvLayer(3, channels[size], 1)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, log_size-num_down, -1):
            out_channel = channels[2 ** (i - 1)]
            stem.append(ResBlock(in_channel, out_channel, downsample=True))
            in_channel = out_channel
        stem += [ResBlock(in_channel, in_channel, downsample=False) for i in range(n_res)]
        self.stem = nn.Sequential(*stem)

        self.content = nn.Sequential(
                        ConvLayer(in_channel, in_channel, 1), 
                        ConvLayer(in_channel, in_channel, 1)
                        )
        style  = []
        for i in range(log_size-num_down, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            style.append(ConvLayer(in_channel, out_channel, 3, downsample=True))
            in_channel = out_channel
        style += [
            nn.Flatten(),
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), 
            EqualLinear(channels[4], latent_dim),
              ]
        self.style = nn.Sequential(*style)


    def forward(self, input):
        act = self.stem(input)
        content = self.content(act)
        style = self.style(act)
        return content, style

class StyleEncoder(nn.Module):
    def __init__(self, size, style_dim, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]
        num_down = 6

        for i in range(log_size, log_size-num_down, -1):
            w = 2 ** (i - 1)
            out_channel = channels[w]
            convs.append(ConvLayer(in_channel, out_channel, 3, downsample=True))
            in_channel = out_channel

        convs += [
                nn.Flatten(),
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], style_dim),
                  ]
        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        style = self.convs(input)
        return style.view(input.size(0), -1)

class LatDiscriminator(nn.Module):
    def __init__(self, style_dim):
        super().__init__()

        fc = [EqualLinear(style_dim, 256, activation='fused_lrelu')]
        for i in range(3):
            fc += [EqualLinear(256, 256, activation='fused_lrelu')]
        fc += [FCMinibatchStd(256, 256)]
        fc += [EqualLinear(256, 1)]
        self.fc = nn.Sequential(*fc)

    def forward(self, input):
        return [self.fc(input), ]

class FCMinibatchStd(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc = EqualLinear(in_channel+1, out_channel, activation='fused_lrelu')

    def forward(self, out):
        stddev = torch.sqrt(out.var(0, unbiased=False) + 1e-8).mean().view(1,1).repeat(out.size(0), 1)
        out = torch.cat([out, stddev], 1)
        out = self.fc(out)
        return out
