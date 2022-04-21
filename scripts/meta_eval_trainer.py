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
from shutil import rmtree

from imageio import imwrite

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from pytorch_fid import fid_score
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from torch_fidelity import calculate_metrics

class MetaEvalTrainer(Trainer):

    def __init__(self, args, eval_class_name, gt_path):
        super().__init__(args)
        self.gt_path = gt_path
        self.eval_class_name = eval_class_name
        self.path_to_tmp_fid = os.path.join(args.path_to_tmp_fid, eval_class_name)
        self.output_dir = os.path.join(self.output_dir, eval_class_name + '_5_shot')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def evaluate_bdd_fid(self):
        self.model.eval()
        if not os.path.exists(self.path_to_tmp_fid):
            os.makedirs(self.path_to_tmp_fid)

        real_path = os.path.join(self.path_to_tmp_fid, 'fid_real/')
        fake_path = os.path.join(self.path_to_tmp_fid, 'fid_fake/')

        # Recreate the folders for fid
        rmtree(real_path, ignore_errors=True)
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(real_path)
        os.makedirs(fake_path)

        num_imgs = 0
        num_imgs_pred = 0
        for batch in self.val_loader:
            batch = [tensor.cuda() for tensor in batch]
            masks = None
            if len(batch) == 6:
                imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
            elif len(batch) == 7:
                imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch

            # Run the model as it has been run during training
            with torch.no_grad():
                model_out = self.model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
            imgs_pred, _, _, _ = model_out
            imgs_pred = imagenet_deprocess_batch(imgs_pred)
            for i in range(imgs_pred.shape[0]):
                img_np = imgs_pred[i].numpy().transpose(1, 2, 0)
                img_path = os.path.join(fake_path, 'img%06d.png' % num_imgs_pred)
                num_imgs_pred += 1
                imwrite(img_path, img_np)
            imgs = imagenet_deprocess_batch(imgs)
            for i in range(imgs.shape[0]):
                img_np = imgs[i].numpy().transpose(1, 2, 0)
                img_path = os.path.join(real_path, 'img%06d.png' % num_imgs)
                num_imgs += 1
                imwrite(img_path, img_np)
            # if num_imgs > 200:
            #     break

        # score = fid_score.calculate_fid_given_paths([real_path, fake_path], 4, 'cuda:0', 2048)
        score = 0
        print("FID score for {} is {}".format(self.eval_class_name, score))
        return score


    def check_folder(self):
        if not os.path.exists(self.path_to_tmp_fid):
            os.makedirs(self.path_to_tmp_fid)

        real_path = self.gt_path
        fake_path = os.path.join(self.path_to_tmp_fid, 'fid_fake/')

        num_imgs = len(os.listdir(real_path))
        alr_ext = 0
        if os.path.exists(fake_path):
            alr_ext = len(os.listdir(fake_path))
        if num_imgs == alr_ext:
            print("Images already exist check passed. No train now")
            return True
        return False

    def evaluate_ag_fid(self):
        self.model.eval()
        if not os.path.exists(self.path_to_tmp_fid):
            os.makedirs(self.path_to_tmp_fid)

        real_path = self.gt_path
        fake_path = os.path.join(self.path_to_tmp_fid, 'fid_fake/')

        num_imgs = len(os.listdir(real_path))
        num_imgs_pred = 0

        alr_ext = 0
        if os.path.exists(fake_path):
            alr_ext = len(os.listdir(fake_path))
        if num_imgs == alr_ext:
            print("Images already exist")
            score = fid_score.calculate_fid_given_paths([real_path, fake_path], 4, 'cuda:0', 2048)
            return score

        # Recreate the folders for fid
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        while num_imgs_pred < num_imgs:
            for batch in self.val_loader:
                batch = [tensor.cuda() for tensor in batch]
                masks = None
                if len(batch) == 6:
                    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
                elif len(batch) == 7:
                    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch

                # Run the model as it has been run during training
                with torch.no_grad():
                    model_out = self.model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
                imgs_pred, _, _, _ = model_out
                imgs_pred = imagenet_deprocess_batch(imgs_pred)
                for i in range(imgs_pred.shape[0]):
                    img_np = imgs_pred[i].numpy().transpose(1, 2, 0)
                    img_path = os.path.join(fake_path, 'img%06d.png' % num_imgs_pred)
                    num_imgs_pred += 1
                    if num_imgs_pred == num_imgs:
                        break
                    imwrite(img_path, img_np)
                if num_imgs_pred == num_imgs:
                    break

        metrics_dict = calculate_metrics(fake_path, real_path, cuda=True, isc=True, fid=True, kid=True, verbose=False,
                                         kid_subset_size=100)
        score = fid_score.calculate_fid_given_paths([real_path, fake_path], 4, 'cuda:0', 2048)
        print("FID score for {} is {}".format(self.eval_class_name, score))
        return score

    def base_train(self, full_train = True):
        num_epochs = 10

        if full_train:
            for i in range(num_epochs):
                for batch in self.train_loader:
                    self.model.train()
                    batch = [tensor.cuda() for tensor in batch]
                    losses, d_obj_losses, d_img_losses = self._run_single_train_iter(batch)
                print("Finished epoch {}".format(i))
        else:
            num_epochs = 80
            for x in self.train_loader:
                batch = x
                break
            for i in range(num_epochs):
                print("Few shot learning is here with {} samples".format(batch[0].shape[0]))
                self.model.train()
                batch = [tensor.cuda() for tensor in batch]
                losses, d_obj_losses, d_img_losses = self._run_single_train_iter(batch)

            # self.print_every_iter(losses, d_obj_losses, d_img_losses)

    def train(self, full_train = True):
        self.base_train(full_train = full_train)
        score = self.evaluate_bdd_fid()
        self.checkpoint_every_iter()
        return score

    def ag_train(self, full_train = True):
        if self.check_folder():
            score = self.evaluate_ag_fid()
            return score
        self.base_train(full_train = full_train)
        score = self.evaluate_ag_fid()
        # self.checkpoint_every_iter()
        return score
