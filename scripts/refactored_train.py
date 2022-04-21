from torch.utils.tensorboard import SummaryWriter
import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from sg2im.data import imagenet_deprocess_batch

from sg2im.discriminators import divide_pred
from sg2im.losses import get_gan_losses, GANLoss
from sg2im.metrics import jaccard
from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager
from sg2im.losses import VGGLoss
from scripts.dataset_loaders import build_loaders
from scripts.model_loaders import build_model, build_obj_discriminator, build_img_discriminator
from scripts.train_utils import add_loss, check_args, check_model, calculate_model_losses

torch.backends.cudnn.benchmark = True
import torchvision

from imageio import imwrite
from shutil import rmtree
from PIL import Image, ImageFile
from pytorch_fid import fid_score
from imageio import imwrite

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Trainer():

    def __init__(self, args):
        check_args(args)

        # Set tensorboard writers
        self.writer = SummaryWriter(args.log_dir + str("/train")) if args.log_dir is not None else None
        self.valWriter = SummaryWriter(args.log_dir + str("/vals")) if args.log_dir is not None else None

        self.num_val_samples = args.num_val_samples

        self.gan_g_loss, self.gan_d_loss = get_gan_losses(args.gan_loss_type)

        self.timing = args.timing

        # Parameters for the discriminator type
        self.use_stylegan_disc = args.use_stylegan_disc
        self.multi_discriminator = args.multi_discriminator

        # Some essential model parameters
        self.spade_gen_blocks = args.spade_gen_blocks
        self.use_crn = args.use_crn
        
        # Loss weights
        self.discriminator_loss_weight = args.discriminator_loss_weight
        self.d_obj_weight = args.d_obj_weight
        self.d_img_weight = args.d_img_weight
        self.ac_loss_weight = args.ac_loss_weight
        self.percept_weight = args.percept_weight
        self.l1_pixel_loss_weight = args.l1_pixel_loss_weight
        self.bbox_pred_loss_weight = args.bbox_pred_loss_weight
        self.predicate_pred_loss_weight = args.predicate_pred_loss_weight
        self.mask_loss_weight = args.mask_loss_weight
        self.weight_change_after_50_iter = args.weight_change_after_50_iter

        # Learning rates
        self.learning_rate = args.learning_rate

        # checkpoint dirs
        self.output_dir = args.output_dir
        self.checkpoint_name = args.checkpoint_name
        self.checkpoint_every = args.checkpoint_every
        self.print_every = args.print_every
        self.eval_mode_after = args.eval_mode_after
        self.max_num_imgs = args.max_num_imgs


        # counter
        self.print_every = args.print_every
        self.num_iterations = args.num_iterations

        # Prepare dataset and models
        self._prepare_dataset(args)
        self._prepare_models(args)

        # FID parameters
        self.path_to_tmp_fid = args.path_to_tmp_fid

    def _prepare_dataset(self, args):
        self.vocab, self.train_loader, self.val_loader = build_loaders(args)

    def _init_models(self, args):
        float_dtype = torch.cuda.FloatTensor
        # Set up generator model
        model, model_kwargs = build_model(args, self.vocab)
        model.type(torch.cuda.FloatTensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # Set up object discriminator
        obj_discriminator, d_obj_kwargs = build_obj_discriminator(args, self.vocab)
        optimizer_d_obj = None
        if obj_discriminator is not None:
            obj_discriminator.type(float_dtype)
            obj_discriminator.train()
            optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(),
                                               lr=args.learning_rate)
        # Set up image discriminator
        img_discriminator, d_img_kwargs = build_img_discriminator(args, self.vocab)
        optimizer_d_img = None
        if img_discriminator is not None:
            img_discriminator.type(float_dtype)
            img_discriminator.train()
            optimizer_d_img = torch.optim.Adam(img_discriminator.parameters(),
                                               lr=args.learning_rate)
        return model, model_kwargs, optimizer, obj_discriminator, d_obj_kwargs, optimizer_d_obj, img_discriminator, d_img_kwargs, optimizer_d_img

    def _prepare_models(self, args):
        model, model_kwargs, optimizer, obj_discriminator, d_obj_kwargs, optimizer_d_obj, \
        img_discriminator, d_img_kwargs, optimizer_d_img = self._init_models(
            args)

        # restore models
        restore_path = None
        if args.checkpoint_start_from is not None:
            restore_path = args.checkpoint_start_from
        else:
            if args.restore_from_checkpoint:
                restore_path = '%s_model.pt' % args.checkpoint_name
                restore_path = os.path.join(args.output_dir, restore_path)
        if restore_path is not None and os.path.isfile(restore_path):
            print('Restoring from checkpoint:{}'.format(restore_path))
            checkpoint = torch.load(restore_path, map_location='cpu')
            self.checkpoint = checkpoint
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optim_state'])

            if obj_discriminator is not None:
                obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
                optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

            if img_discriminator is not None:
                img_discriminator.load_state_dict(checkpoint['d_img_state'])
                optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

            self.t = checkpoint['counters']['t']
            if 0 <= self.eval_mode_after <= self.t:
                model.eval()
            else:
                model.train()
            self.epoch = checkpoint['counters']['epoch']
        else:
            self.t, self.epoch = 0, 0
            self.checkpoint = {
                'args': args.__dict__,
                'vocab': self.vocab,
                'model_kwargs': model_kwargs,
                'd_obj_kwargs': d_obj_kwargs,
                'd_img_kwargs': d_img_kwargs,
                'losses_ts': [],
                'losses': defaultdict(list),
                'd_losses': defaultdict(list),
                'checkpoint_ts': [],
                'train_batch_data': [],
                'train_samples': [],
                'train_iou': [],
                'val_batch_data': [],
                'val_samples': [],
                'val_losses': defaultdict(list),
                'val_iou': [],
                'norm_d': [],
                'norm_g': [],
                'counters': {
                    't': None,
                    'epoch': None,
                },
                'model_state': None, 'model_best_state': None, 'optim_state': None,
                'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
                'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
                'best_t': [],
            }

        # Set up models for training with restored/initialized parameters
        self.obj_discriminator = obj_discriminator
        self.optimizer_d_obj = optimizer_d_obj
        self.img_discriminator = img_discriminator
        self.optimizer_d_img = optimizer_d_img
        self.model = model
        self.optimizer = optimizer

    def _run_single_train_iter(self, batch):
        masks = None
        if len(batch) == 6:
            imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
        elif len(batch) == 7:
            imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
        else:
            assert False
        predicates = triples[:, 1]
        with timeit('forward', self.timing):
            model_boxes = boxes
            model_masks = masks
            model_out = self.model(objs, triples, obj_to_img,
                                   boxes_gt=model_boxes, masks_gt=model_masks)
            imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out

        # Calculate generator losses
        with timeit('loss', self.timing):
            # Skip the pixel loss if using GT boxes
            skip_pixel_loss = (model_boxes is None)
            total_loss, losses = calculate_model_losses(skip_pixel_loss, self.model, imgs, imgs_pred,
                                                        boxes, boxes_pred, masks, masks_pred,
                                                        predicates, predicate_scores, self.l1_pixel_loss_weight,
                                                        self.bbox_pred_loss_weight,
                                                        self.predicate_pred_loss_weight, self.mask_loss_weight)

        if self.obj_discriminator is not None:
            scores_fake, ac_loss = self.obj_discriminator(imgs_pred, objs, boxes, obj_to_img)
            total_loss = add_loss(total_loss, ac_loss, losses, 'ac_loss',
                                  self.ac_loss_weight)
            weight = self.discriminator_loss_weight * self.d_obj_weight
            total_loss = add_loss(total_loss, self.gan_g_loss(scores_fake), losses,
                                  'g_gan_obj_loss', weight)

        if self.img_discriminator is not None:
            if self.multi_discriminator:
                fake_and_real = torch.cat([imgs_pred, imgs], dim=0)
                discriminator_out = self.img_discriminator(fake_and_real)
                scores_fake, scores_real = divide_pred(discriminator_out)

                weight = self.discriminator_loss_weight * self.d_img_weight
                criterionGAN = GANLoss()
                img_g_loss = criterionGAN(scores_fake, True, for_discriminator=False)
                total_loss = add_loss(total_loss, img_g_loss, losses,
                                      'g_gan_img_loss', weight)
            elif self.use_stylegan_disc:
                scores_fake, _ = self.img_discriminator(imgs_pred)
                weight = self.discriminator_loss_weight * self.d_img_weight
                loss = scores_fake.mean()
                total_loss = add_loss(total_loss, loss, losses,
                                      'g_gan_img_loss', weight)
            else:
                scores_fake = self.img_discriminator(imgs_pred)
                weight = self.discriminator_loss_weight * self.d_img_weight
                total_loss = add_loss(total_loss, self.gan_g_loss(scores_fake), losses,
                                      'g_gan_img_loss', weight)
            if self.percept_weight != 0:
                criterionVGG = VGGLoss()
                percept_loss = criterionVGG(imgs_pred, imgs)

                total_loss = add_loss(total_loss, percept_loss, losses,
                                      'g_VGG', self.percept_weight)

        losses['total_loss'] = total_loss.item()

        if not math.isfinite(losses['total_loss']):
            raise ValueError("Got loss = NaN, not backpropping")

        self.optimizer.zero_grad()
        with timeit('backward', self.timing):
            total_loss.backward()
        self.optimizer.step()

        results = []
        results.append(losses)

        # Training discriminator
        if self.obj_discriminator is not None:
            d_obj_losses = LossManager()
            imgs_fake = imgs_pred.detach()
            scores_fake, ac_loss_fake = self.obj_discriminator(imgs_fake, objs, boxes, obj_to_img)
            scores_real, ac_loss_real = self.obj_discriminator(imgs, objs, boxes, obj_to_img)

            d_obj_gan_loss = self.gan_d_loss(scores_real, scores_fake)
            d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
            d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
            d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

            self.optimizer_d_obj.zero_grad()
            d_obj_losses.total_loss.backward()
            self.optimizer_d_obj.step()
            results.append(d_obj_losses)

        if self.img_discriminator is not None:
            d_img_losses = LossManager()
            imgs_fake = imgs_pred.detach()

            if self.multi_discriminator:
                fake_and_real = torch.cat([imgs_fake, imgs], dim=0)
                discriminator_out = self.img_discriminator(fake_and_real)
                scores_fake, scores_real = divide_pred(discriminator_out)

                d_img_gan_loss = criterionGAN(scores_fake, False, for_discriminator=True) \
                                 + criterionGAN(scores_real, True, for_discriminator=True)

                d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')
            elif self.use_stylegan_disc:
                scores_fake, _ = self.img_discriminatoriscriminator(imgs_fake)
                scores_real, _ = self.img_discriminator(imgs)
                divergence = (F.relu(1 + scores_real) + F.relu(1 - scores_fake)).mean()
                disc_loss = divergence
                d_img_losses.add_loss(disc_loss, 'd_img_gan_loss')
            else:
                scores_fake = self.img_discriminator(imgs_fake)
                scores_real = self.img_discriminator(imgs)

                d_img_gan_loss = self.gan_d_loss(scores_real, scores_fake)
                d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')

            self.optimizer_d_img.zero_grad()
            d_img_losses.total_loss.backward()
            self.optimizer_d_img.step()
            results.append(d_img_losses)
        return results

    def print_every_iter(self, losses, d_obj_losses, d_img_losses):
        print('t = %d / %d' % (self.t, self.num_iterations))
        for name, val in losses.items():
            print(' G [%s]: %.4f' % (name, val))
            self.writer.add_scalar('Loss/train/G' + name, val, self.t)

        if self.obj_discriminator is not None:
            for name, val in d_obj_losses.items():
                print(' D_obj [%s]: %.4f' % (name, val))
                self.writer.add_scalar('Loss/train/D' + name, val, self.t)

        if self.img_discriminator is not None:
            for name, val in d_img_losses.items():
                print(' D_img [%s]: %.4f' % (name, val))
                self.writer.add_scalar('Loss/train/D' + name, val, self.t)

    def checkpoint_every_iter(self, save_every=10000, save_every_after=30000):
        self.checkpoint['model_state'] = self.model.state_dict()

        if self.obj_discriminator is not None:
            self.checkpoint['d_obj_state'] = self.obj_discriminator.state_dict()
            self.checkpoint['d_obj_optim_state'] = self.optimizer_d_obj.state_dict()

        if self.img_discriminator is not None:
            self.checkpoint['d_img_state'] = self.img_discriminator.state_dict()
            self.checkpoint['d_img_optim_state'] = self.optimizer_d_img.state_dict()

        self.checkpoint['optim_state'] = self.optimizer.state_dict()
        self.checkpoint['counters']['t'] = self.t
        self.checkpoint['counters']['epoch'] = self.epoch
        checkpoint_path_step = os.path.join(self.output_dir,
                                            '%s_%s_model.pt' % (self.checkpoint_name, str(self.t // save_every)))
        checkpoint_path_latest = os.path.join(self.output_dir,
                                              '%s_model.pt' % (self.checkpoint_name))

        print('Saving checkpoint to ', checkpoint_path_latest)
        torch.save(self.checkpoint, checkpoint_path_latest)
        if self.t % save_every == 0 and self.t >= save_every_after:
            torch.save(self.checkpoint, checkpoint_path_step)

    def calculate_fid(self, real_imgs, fake_imgs):
        if not os.path.exists(self.path_to_tmp_fid):
            os.makedirs(self.path_to_tmp_fid)
        real_path = os.path.join(self.path_to_tmp_fid, 'fid_real/')
        fake_path = os.path.join(self.path_to_tmp_fid, 'fid_fake/')

        # Recreate the folders for fid
        rmtree(real_path, ignore_errors=True)
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(real_path)
        os.makedirs(fake_path)

        for k in range(len(real_imgs)):
            img_np = real_imgs[k, :, :, :].numpy().transpose(1, 2, 0)
            img_path = real_path + '{}.png'.format(k)
            imwrite(img_path, img_np)

        for k in range(len(fake_imgs)):
            img_np = fake_imgs[k, :, :, :].numpy().transpose(1, 2, 0)
            img_path = fake_path + '{}.png'.format(k)
            imwrite(img_path, img_np)
        return fid_score.calculate_fid_given_paths([real_path, fake_path], 4, 'cpu', 2048)

    def validate_on_dataset(self, loader, writer, tb_name = 'Train samples'):
        train_results = check_model(loader, self.model, self.l1_pixel_loss_weight, self.bbox_pred_loss_weight,
                                    self.predicate_pred_loss_weight, self.mask_loss_weight, self.num_val_samples)
        t_losses, t_samples, t_batch_data, t_avg_iou = train_results
        train_fid = self.calculate_fid(t_samples['gt_img'], t_samples['pred_box_pred_mask'])
        writer.add_scalar('FID', train_fid, self.t)

        # write images to tensorboard
        train_samples_viz = torch.cat((t_samples['gt_img'][:self.max_num_imgs, :, :, :],
                                       t_samples['gt_box_gt_mask'][:self.max_num_imgs, :, :, :],
                                       t_samples['pred_box_pred_mask'][:self.max_num_imgs, :, :, :]), dim=3)

        writer.add_image(tb_name, make_grid(train_samples_viz, nrow=4, padding=4), global_step=self.t)
        writer.add_scalar('IOU', t_avg_iou, self.t)

    def validate(self):
        print('checking on train')
        self.validate_on_dataset(self.train_loader, self.writer)
        print('checking on val')
        self.validate_on_dataset(self.val_loader, self.valWriter, tb_name = "Val samples")

    def train(self):
        while True:
            if self.t >= self.num_iterations:
                break
            self.epoch += 1
            print('Starting epoch %d' % self.epoch)

            for batch in self.train_loader:
                self.model.train()
                if self.t >= self.eval_mode_after:
                    self.model.eval()
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                self.t += 1
                batch = [tensor.cuda() for tensor in batch]
                losses, d_obj_losses, d_img_losses = self._run_single_train_iter(batch)

                # Exponentially moving averages
                if not self.spade_gen_blocks and not self.use_crn:
                    # Exponential moving averages
                    if self.t % 10 == 0 and self.t > 20000:
                        self.model.styleGAN.EMA()

                    if self.t <= 25000 and self.t % 1000 == 2:
                        self.model.styleGAN.reset_parameter_averaging()

                if self.weight_change_after_50_iter and self.t == 50000:
                    self.l1_pixel_loss_weight /= 2
                    self.percept_weight *= 1.6

                if self.t % self.print_every == 0:
                    self.print_every_iter(losses, d_obj_losses, d_img_losses)

                if self.t % self.checkpoint_every == 0:
                    self.validate()
                    self.checkpoint_every_iter()
