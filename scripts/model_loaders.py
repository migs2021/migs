from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator, MultiscaleDiscriminator, divide_pred
from sg2im.stylegan2 import Discriminator
from sg2im.model import Sg2ImModel
import os

def build_model(args, vocab):
    if args.checkpoint_start_from is not None:
        checkpoint = torch.load(args.checkpoint_start_from)
        kwargs = checkpoint['model_kwargs']
        model = Sg2ImModel(**kwargs)
        raw_state_dict = checkpoint['model_state']
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        kwargs = {
            'vocab': vocab,
            'image_size': args.image_size,
            "layout_size": args.layout_size,
            'embedding_dim': args.embedding_dim,
            'gconv_dim': args.gconv_dim,
            'gconv_hidden_dim': args.gconv_hidden_dim,
            'gconv_num_layers': args.gconv_num_layers,
            'mlp_normalization': args.mlp_normalization,
            'decoder_dims': args.decoder_dims,
            'normalization': args.normalization,
            'activation': args.activation,
            'mask_size': args.mask_size,
            'layout_noise_dim': args.layout_noise_dim,
            'rank': args.gpu,
            'spade_blocks': args.spade_gen_blocks,
            'stylegan_use_random_noise': args.stylegan_use_random_noise,
            'stylegan_latent_dim': args.stylegan_latent_dim,
            'filters_from_capacity': args.stylegan_filters_from_capacity,
            'use_noisy_layout': args.stylegan_use_noisy_layout,
            'network_capacity': args.stylegan_network_capacity,
            'add_non_up': args.stylegan_add_non_up,
            'init_chan_user_inp': args.stylegan_init_chan_user_inp,
            'use_crn': args.use_crn
        }
        model = Sg2ImModel(**kwargs)
    return model, kwargs


def build_obj_discriminator(args, vocab):
    discriminator = None
    d_kwargs = {}
    d_weight = args.discriminator_loss_weight
    d_obj_weight = args.d_obj_weight
    if d_weight == 0 or d_obj_weight == 0:
        return discriminator, d_kwargs

    d_kwargs = {
        'vocab': vocab,
        'arch': args.d_obj_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
        'object_size': args.crop_size,
    }
    discriminator = AcCropDiscriminator(**d_kwargs)
    return discriminator, d_kwargs


def build_img_discriminator(args, vocab):
    discriminator = None
    d_kwargs = {}
    d_weight = args.discriminator_loss_weight
    d_img_weight = args.d_img_weight
    if d_weight == 0 or d_img_weight == 0:
        return discriminator, d_kwargs

    d_kwargs = {
        'arch': args.d_img_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
    }
    if args.multi_discriminator:
        discriminator = MultiscaleDiscriminator(input_nc=3, num_D=args.multi_discr_dim)
    elif args.use_stylegan_disc:
        discriminator = Discriminator(args.image_size)
    else:
        discriminator = PatchDiscriminator(**d_kwargs)
    return discriminator, d_kwargs
