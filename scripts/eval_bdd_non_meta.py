import yaml
from addict import Dict
from scripts.args_parser import argument_parser
# from scripts.train import main as train_main
from scripts.refactored_train import Trainer
from scripts.meta_eval_trainer import MetaEvalTrainer
import torch
import shutil
import os
import random
import sys
from scripts.dataset_loaders import get_bdd_loaders_from_folder
from sg2im.data.utils import imagenet_deprocess_batch
from imageio import imwrite
from shutil import rmtree
from pytorch_fid import fid_score

from torch_fidelity import calculate_metrics


# needed to solve tuple reading issue
class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)


def main():
    # argument: name of yaml config file
    try:
        filename = sys.argv[1]
    except IndexError as error:
        print("provide yaml file as command-line argument!")
        exit()

    config = Dict(yaml.load(open(filename), Loader=PrettySafeLoader))
    os.makedirs(config.log_dir, exist_ok=True)
    # save a copy in the experiment dir
    shutil.copyfile(filename, os.path.join(config.log_dir, 'args.yaml'))

    torch.cuda.set_device(config.gpu)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    args = yaml_to_parser(config)

    train_path = os.path.join("/media/azadef/MyHDD/sabrina/thesis", 'labels_coco', "bdd100k_labels_images_val_coco.json")
    fid_scores = {}

    few_shot_learning = False
    gt_path = os.path.join("/media/azadef/MyHDD/sabrina/thesis", 'bdd_test', 'default')
    trainer = MetaEvalTrainer(args, "default", gt_path)
    trainer.eval()
    path_all_fake = os.path.join(args.path_to_tmp_fid, 'default', 'fid_fake')
    path_all_real = os.path.join(args.path_to_tmp_fid, 'default', 'fid_real')
    metrics = calculate_metrics(path_all_fake, path_all_real, cuda=False, isc=True, fid=True, kid=True, verbose=False,
                                kid_subset_size=100)
    print(metrics)


def main_tmp():
    # argument: name of yaml config file
    try:
        filename = sys.argv[1]
    except IndexError as error:
        print("provide yaml file as command-line argument!")
        exit()

    config = Dict(yaml.load(open(filename), Loader=PrettySafeLoader))
    os.makedirs(config.log_dir, exist_ok=True)
    # save a copy in the experiment dir
    shutil.copyfile(filename, os.path.join(config.log_dir, 'args.yaml'))

    torch.cuda.set_device(config.gpu)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    args = yaml_to_parser(config)
    val_path = os.path.join("/media/azadef/MyHDD/sabrina/thesis", 'bdd_test', "test")
    _, val_loaders = get_bdd_loaders_from_folder(args, val_path)

    cur_l = 0
    for vl_ld in val_loaders:
        total_l = 0
        output_dir = os.path.join("/media/azadef/MyHDD/sabrina/thesis", 'bdd_test', "test_imgs{}".format(cur_l))
        cur_l += 1
        for batch in vl_ld:
            imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
            imgs = imagenet_deprocess_batch(imgs)

            # Save the generated images
            for i in range(imgs.shape[0]):
                img_np = imgs[i].numpy().transpose(1, 2, 0)
                img_path = os.path.join(output_dir, 'img%06d.png' % (total_l + i))
                imwrite(img_path, img_np)
            total_l += imgs.shape[0]


def yaml_to_parser(config):
    parser = argument_parser()
    args, unknown = parser.parse_known_args()

    args_dict = vars(args)
    for key, value in config.items():
        try:
            args_dict[key] = value
        except KeyError:
            print(key, ' was not found in arguments')
    return args


if __name__ == '__main__':
    main()
