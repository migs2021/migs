from torch.utils.data import DataLoader
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn

def build_dset_nopairs(args, checkpoint):

  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
  }
  dset = VgSceneGraphDataset(**dset_kwargs)

  return dset

def build_eval_loader(args, checkpoint, vocab_t=None, no_gt=False):

    dset = build_dset_nopairs(args, checkpoint)
    collate_fn = vg_collate_fn

    loader_kwargs = {
    'batch_size': 1,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
    }
    loader = DataLoader(dset, **loader_kwargs)

    return loader

