import numpy as np
import math
import torch
from torch import nn
from math import floor, log2
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
#from kornia.filters import filter2D
from functools import partial
#from linear_attention_transformer import ImageLinearAttention
#from vector_quantize_pytorch import VectorQuantize

import torchvision
from torchvision import transforms

EPS = 1e-8


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def exists(val):
    return val is not None


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries=True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])


def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)


def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)
        # print("X shape: " + str(x.shape))

        if exists(prev_rgb):
            # print("prev_rgb shape: " + str(prev_rgb.shape))
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class InputImageVectorizer(nn.Module):
    def __init__(self, emb_depth, layout_size, latent_dim):
        super().__init__()
        # # From 128*128 -> 64*64
        # self.conv1 = nn.Conv2d(emb_depth, emb_depth, 4, stride=2, padding=1)
        # # From 64*64 -> 32*32
        # self.conv2 = nn.Conv2d(emb_depth, emb_depth, 4, stride = 2, padding=1)
        # self.conv3 = nn.Conv2d(emb_depth, 1, 1)
        # self.fc1 = nn.Linear(1024, 1024)
        # -----
        # From 128*128 -> 64*64
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(emb_depth, 128, 3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # # From 64*64 -> 32*32
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        # self.conv3 = nn.Conv2d(64, 1, 1)
        # self.fc1 = nn.Linear(1024, 512)
        # ----
        # From 32*32 -> 16*16
        h, w = layout_size
        self.conv1 = nn.Conv2d(emb_depth, emb_depth, 4, stride=2, padding=1)
        h_new = math.floor((h - 2) / 2) + 1
        w_new = math.floor((w - 2) / 2) + 1
        self.conv2 = nn.Conv2d(emb_depth, 1, 1)
        input_dim = h_new * w_new
        self.fc1 = nn.Linear(input_dim, latent_dim)

        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = leaky_relu()(self.conv1(x))
        x = leaky_relu()(self.conv2(x))
        # x = (self.conv3(x))
        x = torch.flatten(x, 1)
        x = leaky_relu()(self.fc1(x))
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = leaky_relu()(self.conv3(x))
        # x = torch.flatten(x, 1)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)
        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 1, 2))

        noise2 = self.to_noise2(inoise).permute((0, 3, 1, 2))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=16, initial_size=4, transparent=False, attn_layers=[],
                 no_const=False, fmap_max=2048, add_non_up=True, filters_from_capacity=False, init_chan_user_inp=False,
                 init_chans=128):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - log2(initial_size) + 1)
        # print(self.num_layers)

        filters = [int(image_size / (2 ** i)) for i in range(self.num_layers)]

        if filters_from_capacity:
            filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]
        print(filters)

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        if init_chan_user_inp:
            init_channels = init_chans
        else:
            filters = [init_channels, *filters]
        print(init_channels)


        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 32, 32)))

        self.initial_conv = nn.Conv2d(init_channels, filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            if add_non_up:
                self.attns.append(attn_fn)
                block = GeneratorBlock(
                    latent_dim,
                    in_chan,
                    out_chan,
                    upsample=not_first,
                    upsample_rgb=False,
                    rgba=transparent
                )
                self.blocks.append(block)
                block2 = GeneratorBlock(
                    latent_dim,
                    out_chan,
                    out_chan,
                    upsample=False,
                    upsample_rgb=not_last,
                    rgba=transparent
                )
                self.blocks.append(block2)
                print("added non-up conv")
            else:
                block = GeneratorBlock(
                    latent_dim,
                    in_chan,
                    out_chan,
                    upsample=not_first,
                    upsample_rgb=not_last,
                    rgba=transparent
                )
                self.blocks.append(block)

        # print(len(self.blocks))
        if add_non_up:
            self.num_layers *= 2
            print("Num of layers will be {}".format(self.num_layers))

    def forward(self, layout, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        # if self.no_const:
        #     avg_style = styles.mean(dim=1)[:, :, None, None]
        #     x = self.to_initial_block(avg_style)
        # else:
        #     x = self.initial_block.expand(batch_size, -1, -1, -1)
        # print(x.shape)
        x = layout
        # print(x.shape)
        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        # print("Blocks: " + str(len(self.blocks)))
        # print("Styles: " + str(len(styles)))
        # print("Attn: " + str(len(self.attns)))
        for style, block, attn in zip(styles, self.blocks, self.attns):
            # print("block start")
            if exists(attn):
                # print("I am dumbo")
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
            # print("block finish")
        # print(rgb.shape)
        return rgb


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding=1, stride=2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss


class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity=16, fq_layers=[], fq_dict_size=256, attn_layers=[],
                 transparent=False, fmap_max=512):
        super().__init__()
        if image_size[0] != image_size[1]:
            if image_size[1] > image_size[0]:
                self.scale_factor = image_size[1] // image_size[0]
            else:
                self.scale_factor = image_size[0] // image_size[1]
        else:
            self.scale_factor = 1

        self.image_size = image_size[0]
        num_layers = int(log2(self.image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(64) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = self.scale_factor * 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, _, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss


def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


class StyleGAN2(nn.Module):
    def __init__(self, image_size, layout_size, latent_dim=256, initial_size=4, fmap_max=2048, style_depth=8,
                 network_capacity=16, transparent=False, steps=1, lr=1e-4, ttur_mult=2, fq_layers=[], fq_dict_size=256,
                 attn_layers=[], no_const=False, lr_mlp=0.1, rank=0, layout_emb_depth=128, noise_emb_depth=0,
                 use_random_noise=False, filters_from_capacity=False, use_noisy_layout=False, add_non_up=True,
                 init_chan_user_inp=False):
        print("random noise: {}".format(use_random_noise))
        super().__init__()
        print("init layout from user {}".format(init_chan_user_inp))

        self.init_chan_user_inp = init_chan_user_inp
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)
        self.emb_depth = layout_emb_depth + noise_emb_depth
        self.layout_emb_depth = layout_emb_depth
        self.imageVectorized = InputImageVectorizer(self.emb_depth, layout_size, latent_dim)
        self.use_random_noise = use_random_noise
        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, initial_size, transparent=transparent,
                           attn_layers=attn_layers, no_const=no_const, fmap_max=fmap_max,
                           filters_from_capacity=filters_from_capacity, add_non_up=add_non_up,
                           init_chan_user_inp=init_chan_user_inp, init_chans=self.emb_depth)

        self.imageVectorizedE = InputImageVectorizer(self.emb_depth, layout_size, latent_dim)
        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, initial_size, transparent=transparent,
                            attn_layers=attn_layers, no_const=no_const, filters_from_capacity=filters_from_capacity,
                            add_non_up=add_non_up, init_chan_user_inp=init_chan_user_inp, init_chans=self.emb_depth)
        self.use_noisy_layout = use_noisy_layout

        if filters_from_capacity:
            self.num_layers = int(log2(image_size) - log2(initial_size) + 1)
            out_emb_size = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1][0]

        else:
            out_emb_size = 128

        print(out_emb_size)
        if use_noisy_layout:
            print('using noisy layout')
            self.one_o_one = nn.Sequential(
                nn.Conv2d(self.emb_depth, out_emb_size, 1, stride=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.one_o_one = nn.Sequential(
                nn.Conv2d(layout_emb_depth, out_emb_size, 1, stride=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True))

        # turn off grad for exponential moving averages
        set_requires_grad(self.imageVectorizedE, False)
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.imageVectorizedE.parameters()) + list(self.G.parameters()) + list(
            self.S.parameters())
        self.G_opt = Adam(generator_params, lr=self.lr, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()
        self.rank = rank
        self.cuda(rank)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.imageVectorizedE, self.imageVectorized)
        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.imageVectorizedE.load_state_dict(self.imageVectorized.state_dict())
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x, layout_noise, image_noise, layout_flex, use_ema_models=False):
        num_layers = self.G.num_layers
        latent_dim = self.G.latent_dim
        batch_size, _, _, _ = x.size()
        layout = torch.cat([x, layout_noise], dim=1)
        if self.use_random_noise:
            styles = noise_list(batch_size, num_layers, latent_dim, device=self.rank)
            if use_ema_models:
                w_space = latent_to_w(self.SE, styles)
            else:
                w_space = latent_to_w(self.S, styles)
            w_styles = styles_def_to_tensor(w_space)
        else:
            if use_ema_models:
                print("Using ema models")
                img_vector = self.imageVectorizedE(layout)
                style_vector = self.SE(img_vector)
            else:
                img_vector = self.imageVectorized(layout)
                style_vector = self.S(img_vector)
            img_vectors = []
            img_vectors.append((style_vector, num_layers))
            w_styles = styles_def_to_tensor(img_vectors)

        if self.init_chan_user_inp:
            layout_inp = x
        else:
            layout_inp = self.one_o_one(x)
        if use_ema_models:
            x = self.GE(layout_inp, w_styles, image_noise)
        else:
            x = self.G(layout_inp, w_styles, image_noise)
        return x
