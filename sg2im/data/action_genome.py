import json, os, random, math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils

from .utils import imagenet_preprocess, Resize


class ActionGenomeSceneGraphDataset(Dataset):
    def __init__(self, image_dir, annotation_json, obj_classes_file, rel_classes_file,
                 image_size=(64, 64), normalize_images=True):
        """
        A PyTorch Dataset for loading Action genome
        """
        super(Dataset, self).__init__()

        self.image_dir = image_dir
        self.normalize_images = normalize_images
        self.set_image_size(image_size)

        self.image_paths = []
        with open(annotation_json, 'r') as f:
            self.annot_data = json.load(f)
            for k, _ in self.annot_data.items():
                self.image_paths.append(k)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
            'pred_idx_to_name': {}
        }

        with open(obj_classes_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.rstrip('\n')
                self.vocab['object_name_to_idx'][line] = idx

        self.vocab['object_name_to_idx']['__image__'] = len(self.vocab['object_name_to_idx'])

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name


        with open(rel_classes_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.rstrip('\n')
                self.vocab['pred_name_to_idx'][line] = idx
                self.vocab['pred_idx_to_name'][idx] = line

        in_image_idx = len(self.vocab['pred_name_to_idx'])
        self.vocab['pred_name_to_idx']['__in_image__'] = in_image_idx

        # Build object_idx_to_name
        name_to_idx = self.vocab['pred_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['pred_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['pred_idx_to_name'] = idx_to_name

        # print(self.vocab)

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_paths):
            num_objs = len(self.annot_data[image_id]['obj_list'])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a scene graph for action genome dataset
        """
        filename = self.image_paths[index]
        annots = self.annot_data[filename]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size
        objs, boxes = [], []
        for idx, obj in enumerate(annots['obj_list']):
            objs.append(self.vocab['object_name_to_idx'][obj])
            boxes.append(torch.FloatTensor(annots['bboxes'][idx]))

        # Add dummy __image__ object
        objs.append(self.vocab['object_name_to_idx']['__image__'])
        boxes.append(torch.FloatTensor([0, 0, 1, 1]))

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)

        # Add triples
        triples = []
        for s, p, o in annots['triplets']:
            p = self.vocab['pred_name_to_idx'][p]
            triples.append([s, p, o])

        # Add __in_image__ triples
        O = objs.size(0)
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        return image, objs, boxes, triples


def ag_collate_fn(batch):
  """
  Collate function to be used when wrapping a ActionGenomeSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (O,) giving categories for all objects
  - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
  - triples: FloatTensor of shape (T, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
  - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
  - triple_to_img: LongTensor of shape (T,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
  """
  # batch is a list, and each element is (image, objs, boxes, triples)
  all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  obj_offset = 0
  for i, (img, objs, boxes, triples) in enumerate(batch):
    all_imgs.append(img[None])
    O, T = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_boxes.append(boxes)
    triples = triples.clone()
    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(O).fill_(i))
    all_triple_to_img.append(torch.LongTensor(T).fill_(i))
    obj_offset += O

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_imgs, all_objs, all_boxes, all_triples,
         all_obj_to_img, all_triple_to_img)
  return out


def ag_uncollate_fn(batch):
  """
  Inverse operation to the above.
  """
  imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
  out = []
  obj_offset = 0
  for i in range(imgs.size(0)):
    cur_img = imgs[i]
    o_idxs = (obj_to_img == i).nonzero().view(-1)
    t_idxs = (triple_to_img == i).nonzero().view(-1)
    cur_objs = objs[o_idxs]
    cur_boxes = boxes[o_idxs]
    cur_triples = triples[t_idxs].clone()
    cur_triples[:, 0] -= obj_offset
    cur_triples[:, 2] -= obj_offset
    obj_offset += cur_objs.size(0)
    out.append((cur_img, cur_objs, cur_boxes, cur_triples))
  return out

