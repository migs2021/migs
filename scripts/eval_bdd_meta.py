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

    train_path = os.path.join("./", 'bdd_top_classes', "train")
    val_path = os.path.join("/./", 'bdd_top_classes', "test")
    fid_scores = {}

    few_shot_learning = False
    all_metrics = {}
    for meta_class in os.listdir(val_path):
        meta_name = meta_class.rstrip('.json')
        args.coco_train_instances_json = os.path.join(train_path, meta_class)
        args.coco_val_instances_json = os.path.join(val_path, meta_class)
        gt_path = os.path.join("./", 'bdd_test', meta_name)
        print(meta_name)
        if few_shot_learning:
            args.batch_size = 5
        trainer = MetaEvalTrainer(args, meta_name, gt_path)
        score = trainer.train(full_train = (not few_shot_learning))
        fid_scores[meta_name] = score
        print('finished with one class')
        print(all_metrics)
    path_to_fid = args.path_to_tmp_fid
    path_all_real = os.path.join(path_to_fid, 'all_real')
    path_all_fake = os.path.join(path_to_fid, 'all_fake')
    # Recreate the folders for fid
    rmtree(path_all_real, ignore_errors=True)
    rmtree(path_all_fake, ignore_errors=True)
    os.makedirs(path_all_real)
    os.makedirs(path_all_fake)


    for key in fid_scores.keys():
        real_class_imgs =  os.path.join(path_to_fid, key, 'fid_real')
        fake_class_imgs = os.path.join(path_to_fid, key, 'fid_fake')
        for f in os.listdir(real_class_imgs):
            class_path = os.path.join(real_class_imgs, f)
            file_name = '{}_{}.png'.format(key, f.rstrip('.png'))
            path_to_all = os.path.join(path_all_real, file_name)
            os.popen('cp {} {}'.format(class_path, path_to_all))

        for f in os.listdir(fake_class_imgs):
            class_path = os.path.join(fake_class_imgs, f)
            file_name = '{}_{}.png'.format(key, f.rstrip('.png'))
            path_to_all = os.path.join(path_all_fake, file_name)
            os.popen('cp {} {}'.format(class_path, path_to_all))
    metrics = calculate_metrics(path_all_fake, path_all_real, cuda=True, isc=True, fid=True, kid=True, verbose=False, kid_subset_size = 100)
    print(fid_scores)
    print(metrics)

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
