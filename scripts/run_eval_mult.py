from scripts.evaluate_reconstruction import main
import argparse
import numpy as np
import torch
import os
import yaml
import tqdm
from addict import Dict
from collections import defaultdict
from sg2im.utils import int_tuple, bool_flag

GPU = 0
EVAL_ALL = True         # evaluate on all bounding boxes (batch size=1)
IGNORE_SMALL = True

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', default='/media/azadef/MyHDD/sabrina/thesis/configs/')
parser.add_argument('--experiment', default="expr_8", type=str)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--dataset', default='vg', choices=['clevr', 'vg'])
parser.add_argument('--with_feats', default=False, type=bool_flag)
parser.add_argument('--generative', default=True, type=bool_flag)
parser.add_argument('--predgraphs', default=False, type=bool_flag)

parser.add_argument('--data_h5', default=None)

parser.add_argument('--image_size', default=(128, 128), type=int_tuple)
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--print_every', default=500, type=int)
parser.add_argument('--save_every', default=500, type=int)

parser.add_argument('--visualize_imgs_boxes', default=False, type=bool_flag)
parser.add_argument('--visualize_graphs', default=False, type=bool_flag)
parser.add_argument('--save_images', default=True, type=bool_flag)
parser.add_argument('--save_gt_images', default=True, type=bool_flag)

args = parser.parse_args()

if args.dataset == "clevr":
    DATA_DIR = "/media/azadef/MyHDD/Data/CLEVR_SIMSG/target/"
    args.data_image_dir = DATA_DIR
else:
    DATA_DIR = "/media/azadef/MyHDD/Data/vg/"
    args.data_image_dir = os.path.join(DATA_DIR, 'images')

if args.data_h5 is None:
    if args.predgraphs:
        args.data_h5 = os.path.join(DATA_DIR, 'test_predgraphs.h5')
    else:
        args.data_h5 = os.path.join(DATA_DIR, 'test.h5')

if args.checkpoint is None:
    ckpt = args.exp_dir + args.experiment
    args.checkpoint = './{}_model.pt'.format(ckpt)

CONFIG_FILE = args.exp_dir + '{}.yaml'.format(args.experiment)
IMG_SAVE_PATH = args.exp_dir + 'logs/{}/evaluation/'.format(args.experiment)
RESULT_SAVE_PATH = args.exp_dir + 'logs/{}/evaluation/results/'.format(args.experiment)
RESULT_FILE = RESULT_SAVE_PATH + '{}/test_results_{}.pickle'

USE_GT_BOXES = True     # use ground truth bounding boxes for evaluation
print("feats", args.with_feats)
torch.cuda.set_device(GPU)
device = torch.device(GPU)

chk_start = 5
chk_fin = 9

if __name__ == '__main__':
    print('hey')
    for i in range(chk_start, chk_fin):
        checkpoint = args.checkpoint + '{}_model.pt'.format(i)
        # print(args.checkpoint)
        print("this is the chkp number {}".format(i))
        main(checkpoint)
        os.system('python -m pytorch_fid --device cuda:0 configs/logs/{}/evaluation/generative/ configs/logs/{}/evaluation/gt'.format(args.experiment, args.experiment))