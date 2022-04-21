from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.bdd import BDDSceneGraphDataset
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.data.action_genome import ActionGenomeSceneGraphDataset, ag_collate_fn
from torch.utils.data import DataLoader, ConcatDataset
import os
import json

def build_coco_dsets(args):
    dset_kwargs = {
        'image_dir': args.coco_train_image_dir,
        'instances_json': args.coco_train_instances_json,
        'stuff_json': args.coco_train_stuff_json,
        'stuff_only': args.coco_stuff_only,
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'max_samples': args.num_train_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
        'include_relationships': args.include_relationships,
        'no_mask': args.coco_no_mask,
    }
    train_dset = CocoSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    dset_kwargs['image_dir'] = args.coco_val_image_dir
    dset_kwargs['instances_json'] = args.coco_val_instances_json
    dset_kwargs['stuff_json'] = args.coco_val_stuff_json
    dset_kwargs['max_samples'] = args.num_val_samples
    val_dset = CocoSceneGraphDataset(**dset_kwargs)

    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))

    return vocab, train_dset, val_dset


def build_bdd_one_class(args, dset_name):
    dset_kwargs = {
        'image_dir': args.coco_train_image_dir,
        'instances_json': dset_name,
        'stuff_json': args.coco_train_stuff_json,
        'stuff_only': args.coco_stuff_only,
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'max_samples': args.num_train_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
        'include_relationships': args.include_relationships,
        'no_mask': args.coco_no_mask,
    }
    train_dset = BDDSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset {} has {} images and {} objects'.format(dset_name, num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))
    vocab = json.loads(json.dumps(train_dset.vocab))

    return vocab, train_dset

def build_bdd_dsets(args):
    dset_kwargs = {
        'image_dir': args.coco_train_image_dir,
        'instances_json': args.coco_train_instances_json,
        'stuff_json': args.coco_train_stuff_json,
        'stuff_only': args.coco_stuff_only,
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'max_samples': args.num_train_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
        'include_relationships': args.include_relationships,
        'no_mask': args.coco_no_mask,
    }
    train_dset = BDDSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    dset_kwargs['image_dir'] = args.coco_val_image_dir
    dset_kwargs['instances_json'] = args.coco_val_instances_json
    dset_kwargs['stuff_json'] = args.coco_val_stuff_json
    dset_kwargs['max_samples'] = args.num_val_samples
    val_dset = BDDSceneGraphDataset(**dset_kwargs)

    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))

    return vocab, train_dset, val_dset

def get_vg_merged_dsets(args):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': args.train_h5,
        'image_dir': args.vg_image_dir,
        'image_size': args.image_size,
        'max_samples': args.num_train_samples,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.vg_use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    dset = VgSceneGraphDataset(**dset_kwargs)

    return vocab, dset

def build_vg_dsets(args):
    train_path = os.path.join(args.meta_base_dst_path, "Meta_h5", "train")
    val_path = os.path.join(args.meta_base_dst_path, "Meta_h5", "val")
    train_dsets = []
    val_dsets = []

    meta_names = range(60)

    for meta_class in meta_names:
        dset_name = os.path.join(train_path, str(meta_class) + ".h5")
        _, dset = build_vg_one_dsets(args, dset_name, False)
        train_dsets.append(dset)

    train_dset = ConcatDataset(train_dsets)


    meta_names = range(60, 100)
    for meta_class in meta_names:
        dset_name = os.path.join(val_path, str(meta_class) + ".h5")
        vocab, dset = build_vg_one_dsets(args, dset_name, True)
        val_dsets.append(dset)

    val_dset = ConcatDataset(val_dsets)

    return vocab, train_dset, val_dset


def build_vg_dsets_def(args):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': args.train_h5,
        'image_dir': args.vg_image_dir,
        'image_size': args.image_size,
        'max_samples': args.num_train_samples,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.vg_use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    train_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['h5_path'] = args.val_h5
    del dset_kwargs['max_samples']
    val_dset = VgSceneGraphDataset(**dset_kwargs)

    return vocab, train_dset, val_dset


def build_ag_dsets(args):
    dset_kwargs = {
        'image_dir': args.ag_image_dir,
        'annotation_json': args.ag_train_annotation_json,
        'obj_classes_file': args.ag_obj_classes_file,
        'rel_classes_file': args.ag_rel_classes_file,
        'image_size': args.image_size,
    }
    train_dset = ActionGenomeSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print("Iterations per epoch {}".format(iter_per_epoch))

    dset_kwargs['annotation_json'] = args.ag_val_annotation_json
    val_dset = ActionGenomeSceneGraphDataset(**dset_kwargs)

    vocab = json.loads(json.dumps(train_dset.vocab))
    return vocab, train_dset, val_dset

def build_ag_one_dsets(args, dset_name):
    dset_kwargs = {
        'image_dir': args.ag_image_dir,
        'annotation_json': dset_name,
        'obj_classes_file': args.ag_obj_classes_file,
        'rel_classes_file': args.ag_rel_classes_file,
        'image_size': args.image_size,
    }
    train_dset = ActionGenomeSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print("Iterations per epoch {}".format(iter_per_epoch))
    vocab = json.loads(json.dumps(train_dset.vocab))
    return vocab, train_dset

def build_vg_one_dsets(args, dset_name, val):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    #h5_path = dset_name + '.h5py'
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': dset_name,
        'image_dir': args.vg_image_dir,
        'image_size': args.image_size,
        'max_samples': args.num_train_samples,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.vg_use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    if val:
        #dset_kwargs['h5_path'] = args.val_h5
        del dset_kwargs['max_samples']

    train_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)


    #val_dset = VgSceneGraphDataset(**dset_kwargs)

    return vocab, train_dset #, val_dset

def build_loaders(args):
    if args.dataset == 'vg':
        vocab, train_dset, val_dset = build_vg_dsets(args)
        collate_fn = vg_collate_fn
    elif args.dataset == 'coco':
        vocab, train_dset, val_dset = build_coco_dsets(args)
        collate_fn = lambda b: coco_collate_fn(b, no_mask=args.coco_no_mask)
    elif args.dataset == 'bdd':
        vocab, train_dset, val_dset = build_bdd_dsets(args)
        collate_fn = lambda b: coco_collate_fn(b, no_mask=args.coco_no_mask)
    elif args.dataset == 'ag':
        vocab, train_dset, val_dset = build_ag_dsets(args)
        collate_fn = ag_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader

def get_bdd_loaders_from_folder(args, path):
    loaders = []
    for meta_class in os.listdir(path):
        dset_name = os.path.join(path, meta_class)
        vocab, dset = build_bdd_one_class(args, dset_name)
        collate_fn = lambda b: coco_collate_fn(b, no_mask=args.coco_no_mask)
        loader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.loader_num_workers,
            'shuffle': True,
            'collate_fn': collate_fn,
        }
        loader = DataLoader(dset, **loader_kwargs)
        loaders.append(loader)
    return vocab, loaders

def get_ag_loaders_from_folder(args, path):
    loaders = []
    for meta_class in os.listdir(path):
        dset_name = os.path.join(path, meta_class)
        vocab, dset = build_ag_one_dsets(args, dset_name)
        collate_fn = ag_collate_fn
        loader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.loader_num_workers,
            'shuffle': True,
            'collate_fn': collate_fn,
        }
        loader = DataLoader(dset, **loader_kwargs)
        loaders.append(loader)
    return vocab, loaders


def get_vg_loaders_from_folder(args, path, val):
    loaders = []
    if val:
        meta_names = range(60,100)
    else:
        meta_names = range(60)
    for meta_class in meta_names:
        dset_name = os.path.join(path, str(meta_class) + ".h5")
        vocab, dset = build_vg_one_dsets(args, dset_name, val)
        collate_fn = vg_collate_fn
        loader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.loader_num_workers,
            'shuffle': True,
            'collate_fn': collate_fn,
        }
        if len(dset) > 0:
            loader = DataLoader(dset, **loader_kwargs)
            loaders.append(loader)
    return vocab, loaders

def build_meta_loaders(args):
    # The assumption is that for each of the meta classes the vocab will be the same
    train_path = os.path.join(args.meta_base_dst_path, "Meta_h5", "train")
    val_path = os.path.join(args.meta_base_dst_path, "Meta_h5", "val")
    if args.dataset == 'bdd':
        vocab, train_loaders = get_bdd_loaders_from_folder(args, train_path)
        _, val_loaders = get_bdd_loaders_from_folder(args, val_path)
        return vocab, train_loaders, val_loaders
    elif args.dataset == 'ag':
        vocab, train_loaders = get_ag_loaders_from_folder(args, train_path)
        _, val_loaders = get_ag_loaders_from_folder(args, val_path)
        return vocab, train_loaders, val_loaders
    elif args.dataset == 'vg':
        print("VG Training set")
        vocab, train_loaders = get_vg_loaders_from_folder(args, train_path, False)
        print("VG validation set")
        _, val_loaders = get_vg_loaders_from_folder(args, val_path, True)
        return vocab, train_loaders, val_loaders



