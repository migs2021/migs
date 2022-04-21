import argparse
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import bool_flag
import os

#VG_DIR = os.path.expanduser('datasets/vg')
VG_DIR = '/media/azadef/MyHDD/Code/GANs/SIMSG/sg2im/datasets/vg'
COCO_DIR = os.path.expanduser('datasets/coco')

def argument_parser():
  # helps parsing the same arguments in a different script
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='coco', choices=['vg', 'coco', 'bdd', 'ag'])
  parser.add_argument('--gpu', default=0, type=int)
  # Optimization hyperparameters
  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--num_iterations', default=1000000, type=int)
  parser.add_argument('--learning_rate', default=1e-4, type=float)

  # Switch the generator to eval mode after this many iterations
  parser.add_argument('--eval_mode_after', default=100000, type=int)

  # Dataset options common to both VG and COCO
  parser.add_argument('--image_size', default='64,64', type=int_tuple)
  parser.add_argument('--layout_size', default=None, type=int_tuple)
  parser.add_argument('--num_train_samples', default=None, type=int)
  parser.add_argument('--num_val_samples', default=1024, type=int)
  parser.add_argument('--shuffle_val', default=True, type=bool_flag)
  parser.add_argument('--loader_num_workers', default=4, type=int)
  parser.add_argument('--include_relationships', default=True, type=bool_flag)

  # VG-specific options
  parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
  parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
  parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
  parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
  parser.add_argument('--max_objects_per_image', default=10, type=int)
  parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

  # COCO-specific options
  parser.add_argument('--coco_train_image_dir',
           default=os.path.join(COCO_DIR, 'images/train2017'))
  parser.add_argument('--coco_val_image_dir',
           default=os.path.join(COCO_DIR, 'images/val2017'))
  parser.add_argument('--coco_train_instances_json',
           default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
  parser.add_argument('--coco_train_stuff_json',
           default=None)
  parser.add_argument('--coco_val_instances_json',
           default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
  parser.add_argument('--coco_val_stuff_json',
           default=None)
  parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
  parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
  parser.add_argument('--coco_include_other', default=False, type=bool_flag)
  parser.add_argument('--min_object_size', default=0.02, type=float)
  parser.add_argument('--min_objects_per_image', default=3, type=int)
  parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)
  parser.add_argument('--coco_no_mask', default=False, type=bool_flag)

  # Action Genome specific options
  parser.add_argument('--ag_image_dir',
           default=os.path.join(COCO_DIR, 'images/train2017'))
  parser.add_argument('--ag_train_annotation_json',
           default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
  parser.add_argument('--ag_val_annotation_json',
           default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
  parser.add_argument('--ag_obj_classes_file',
                      default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
  parser.add_argument('--ag_rel_classes_file',
                      default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))

  # Generator options
  parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
  parser.add_argument('--embedding_dim', default=128, type=int)
  parser.add_argument('--gconv_dim', default=128, type=int)
  parser.add_argument('--gconv_hidden_dim', default=512, type=int)
  parser.add_argument('--gconv_num_layers', default=5, type=int)
  parser.add_argument('--mlp_normalization', default='none', type=str)
  parser.add_argument('--decoder_dims', default='1024,512,256,128,64', type=int_tuple)
  parser.add_argument('--normalization', default='batch')
  parser.add_argument('--activation', default='leakyrelu-0.2')
  parser.add_argument('--layout_noise_dim', default=32, type=int)
  parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

  # Generator losses
  parser.add_argument('--mask_loss_weight', default=0, type=float)
  parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
  parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
  parser.add_argument('--predicate_pred_loss_weight', default=0, type=float) # DEPRECATED

  # Generic discriminator options
  parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
  parser.add_argument('--gan_loss_type', default='gan')
  parser.add_argument('--d_clip', default=None, type=float)
  parser.add_argument('--d_normalization', default='batch')
  parser.add_argument('--d_padding', default='valid')
  parser.add_argument('--d_activation', default='leakyrelu-0.2')

  # Object discriminator
  parser.add_argument('--d_obj_arch',
      default='C4-64-2,C4-128-2,C4-256-2')
  parser.add_argument('--crop_size', default=32, type=int)
  parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight
  parser.add_argument('--ac_loss_weight', default=0.1, type=float)

  # Image discriminator
  parser.add_argument('--d_img_arch',
      default='C4-64-2,C4-128-2,C4-256-2')
  parser.add_argument('--d_img_weight', default=1.0, type=float) # multiplied by d_loss_weight

  # Output options
  parser.add_argument('--print_every', default=10, type=int)
  parser.add_argument('--timing', default=False, type=bool_flag)
  parser.add_argument('--checkpoint_every', default=200, type=int)
  parser.add_argument('--eval_every', default=200, type=int)
  parser.add_argument('--output_dir', default=os.getcwd())
  parser.add_argument('--checkpoint_name', default='checkpoint')
  parser.add_argument('--checkpoint_start_from', default=None)
  parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)

  # tensorboard options
  parser.add_argument('--log_dir', default="./runs/expr_0000", type=str)
  parser.add_argument('--max_num_imgs', default=None, type=int)
  parser.add_argument('--path_to_tmp_fid', default="./fid/expr_0000", type=str)

  # losses weights
  parser.add_argument('--percept_weight', default=0., type=float)
  parser.add_argument('--weight_change_after_50_iter', default=False, type=bool_flag)

  # Spade arguments
  parser.add_argument('--spade_gen_blocks', default=False, type=bool_flag)
  parser.add_argument('--multi_discriminator', default=False, type=bool_flag)
  parser.add_argument('--multi_discr_dim', default=2, type=int)

  # StyleGan arguments
  parser.add_argument('--stylegan_use_random_noise', default=False, type=bool_flag)
  parser.add_argument('--stylegan_latent_dim', default=256, type=int)
  parser.add_argument('--use_stylegan_disc', default=False, type=bool_flag)
  parser.add_argument('--stylegan_filters_from_capacity', default=False, type=bool_flag)
  parser.add_argument('--stylegan_use_noisy_layout', default=False, type=bool_flag)
  parser.add_argument('--stylegan_network_capacity', default=16, type=int)
  parser.add_argument('--stylegan_add_non_up', default=True, type=bool_flag)
  parser.add_argument('--stylegan_init_chan_user_inp', default=False, type=bool_flag)
  
  #CRN generator
  parser.add_argument('--use_crn', default=False, type=bool_flag)

  # Meta learning parameters
  parser.add_argument('--meta_learning_rate', default=1e-4, type=float)
  parser.add_argument('--meta_inner_epochs', default=3, type=int)
  return parser
