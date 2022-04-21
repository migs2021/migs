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
from scripts.dataset_loaders import build_meta_loaders
from scripts.model_loaders import build_model, build_obj_discriminator, build_img_discriminator
from scripts.train_utils import add_loss, check_args, check_model, calculate_model_losses
from scripts.refactored_train import Trainer

torch.backends.cudnn.benchmark = True
import torchvision

from imageio import imwrite

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class MetaTrainer(Trainer):

    def __init__(self, args):
        super().__init__(args)
        self.meta_learning_rate = args.meta_learning_rate
        self.inner_epochs = args.meta_inner_epochs
        self.eval_every = args.eval_every

    def make_infinite(self, dataloader):
        while True:
            for x in dataloader:
                yield x


    def _prepare_dataset(self, args):
        self.vocab, self.train_loaders, self.val_loaders = build_meta_loaders(args)
        if len(self.train_loaders) > 30:
            self.too_many_tasks = True
        else:
            self.too_many_tasks = False
        if not self.too_many_tasks:
            train_iters = []
            for loader in self.train_loaders:
                train_iters.append(self.make_infinite(loader))
            self.train_iters = train_iters
            val_iters = []
            for loader in self.val_loaders:
                val_iters.append(self.make_infinite(loader))
            self.val_iters = val_iters


    def _prepare_models(self, args):
        super()._prepare_models(args)

        # set learning rate of outer loop to be different
        args.learning_rate = args.meta_learning_rate
        # Set up meta model, meaning the ones that are updated in the outer loop
        model, model_kwargs, optimizer, obj_discriminator, d_obj_kwargs, optimizer_d_obj, \
        img_discriminator, d_img_kwargs, optimizer_d_img = self._init_models(
            args)

        model.load_state_dict(self.model.state_dict())
        model.to('cpu')
        # optimizer.load_state_dict(self.optimizer.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=args.meta_learning_rate)
        obj_discriminator.load_state_dict(self.obj_discriminator.state_dict())
        obj_discriminator.to('cpu')
        optimizer_d_obj = torch.optim.SGD(obj_discriminator.parameters(), lr=args.meta_learning_rate)
        # optimizer_d_obj.load_state_dict(self.optimizer_d_obj.state_dict())
        img_discriminator.load_state_dict(self.img_discriminator.state_dict())
        img_discriminator.to('cpu')
        optimizer_d_img = torch.optim.SGD(img_discriminator.parameters(), lr=args.meta_learning_rate)
        # optimizer_d_img.load_state_dict(self.optimizer_d_img.state_dict())


        self.meta_model = model.cpu()
        self.meta_optimizer = optimizer
        self.meta_obj_discriminator = obj_discriminator.cpu()
        self.meta_optimizer_d_obj = optimizer_d_obj
        self.meta_img_discriminator = img_discriminator.cpu()
        self.meta_optimizer_d_img = optimizer_d_img

    def _accumulate_losses(self, losses, losses_to_add):
        #TODO rework it
        if losses is None:
            return losses_to_add
        for name, val in losses_to_add.items():
            if name not in losses.items():
                losses[name] = val
            else:
                losses[name] += val
        return losses

    def _get_average_loss(self, losses, num_iter):
        for name, val in losses.items():
            losses[name] /= num_iter
        return losses

    def meta_training_loop(self):
        task = random.randrange(len(self.train_loaders) - 1)

        if self.too_many_tasks:
            task_loader = self.train_loaders[task]
            batches = []
            for x in task_loader:
                batches.append(x)
            task_len = len(batches)
            cur_task_len = 0
        else:
            task_loader = self.train_iters[task]
        print("Picked a task nummber {}".format(task))
        losses = None
        d_obj_losses = None
        d_img_losses = None
        num_of_iters = 0
        for i in range(self.inner_epochs):
            print("Starting inner epoch {}".format(i))
            if self.too_many_tasks:
                print("Too many tasks, reducing the data")
                batch = batches[cur_task_len]
                cur_task_len += 1
                if cur_task_len == task_len:
                    cur_task_len = 0
            else:
                batch = next(task_loader)
            batch = [tensor.cuda() for tensor in batch]
            losses, d_obj_losses, d_img_losses = self._run_single_train_iter(batch)
            # losses = self._accumulate_losses(losses, losses_to_add)
            # d_obj_losses = self._accumulate_losses(d_obj_losses, d_obj_losses_to_add)
            # d_img_losses = self._accumulate_losses(d_img_losses, d_img_losses_to_add)
            num_of_iters += 1
            if num_of_iters % 5 == 0:
                self.print_every_iter(losses, d_obj_losses, d_img_losses)

        # losses = _get_average_loss(losses, num_of_iters)
        # d_obj_losses = _get_average_loss(d_obj_losses, num_of_iters)
        # d_img_losses = _get_average_loss(d_img_losses, num_of_iters)

        # Updating model generator
        for meta_p, p in zip(self.meta_model.parameters(), self.model.parameters()):
            diff = meta_p - p.cpu()
            meta_p.grad = diff
        self.meta_optimizer.step()

        # Updating obj discriminator
        for meta_p, p in zip(self.meta_obj_discriminator.parameters(), self.obj_discriminator.parameters()):
            diff = meta_p - p.cpu()
            meta_p.grad = diff
        self.meta_optimizer_d_obj.step()

        # Updating img discriminator
        for meta_p, p in zip(self.meta_img_discriminator.parameters(), self.img_discriminator.parameters()):
            diff = meta_p.cpu() - p.cpu()
            meta_p.grad = diff
        self.meta_optimizer_d_img.step()

        return losses, d_obj_losses, d_img_losses

    def reset_meta_model(self):
        self.model.train()
        self.obj_discriminator.train()
        self.img_discriminator.train()
        self.model.load_state_dict(self.meta_model.state_dict())
        self.obj_discriminator.load_state_dict(self.meta_obj_discriminator.state_dict())
        self.img_discriminator.load_state_dict(self.meta_img_discriminator.state_dict())

        if self.t >= self.eval_mode_after:
            self.model.eval()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.meta_model.eval()
            # self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=self.meta_learning_rate)

    def validate(self):
        print('checking on train')
        self.reset_meta_model()
        task = random.randrange(len(self.train_loaders) - 1)
        train_loader = self.train_loaders[task]
        
        num_of_iters = 0
        for _ in range(self.inner_epochs):
            for batch in train_loader:
                batch = [tensor.cuda() for tensor in batch]
                batch_size = len(batch)
                losses_to_add, d_obj_losses_to_add, d_img_losses_to_add = self._run_single_train_iter(batch)
                num_of_iters += 1
                if num_of_iters*batch_size >= 1000:
                    break

        self.validate_on_dataset(train_loader, self.writer)
        self.reset_meta_model()

        print('checking on val')
        task = random.randrange(len(self.val_loaders) - 1)
        val_loader = self.val_loaders[task]
        num_of_iters = 0
        for _ in range(self.inner_epochs):
            for batch in val_loader:
                batch = [tensor.cuda() for tensor in batch]
                batch_size = len(batch)
                losses_to_add, d_obj_losses_to_add, d_img_losses_to_add = self._run_single_train_iter(batch)
                num_of_iters += 1
                if num_of_iters * batch_size >= 1000:
                    break
        self.validate_on_dataset(val_loader, self.valWriter, tb_name = "Val samples")

    def train(self):
        # TODO: do something with EMA
        while True:
            if self.t >= self.num_iterations:
                break
            self.t += 1
            self.reset_meta_model()
            losses, d_obj_losses, d_img_losses = self.meta_training_loop()
            if self.t % self.print_every == 0:
                self.print_every_iter(losses, d_obj_losses, d_img_losses)

            if self.t % self.checkpoint_every == 0:
                self.reset_meta_model()
                self.checkpoint_every_iter(save_every=3000, save_every_after=9000)
            if self.t % self.eval_every == 0:
                self.reset_meta_model()
                self.validate()
