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

from sg2im.metrics import jaccard
from sg2im.data import imagenet_deprocess_batch

def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def check_args(args):
    H, W = args.image_size
    for _ in args.decoder_dims[1:]:
        H = H // 2
    if H == 0:
        raise ValueError("Too many layers in refinement network")


def check_model(loader, model, l1_pixel_weight, bbox_pred_loss_weight,
                predicate_pred_loss_weight, mask_loss_weight, num_val_samples):
    model.eval()
    num_samples = 0
    all_losses = defaultdict(list)
    total_iou = 0
    total_boxes = 0
    with torch.no_grad():
        for batch in loader:

            batch = [tensor.cuda() for tensor in batch]
            masks = None
            if len(batch) == 6:
                imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
            elif len(batch) == 7:
                imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
            predicates = triples[:, 1]

            # Run the model as it has been run during training
            model_masks = masks
            model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
            imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out

            skip_pixel_loss = False
            total_loss, losses = calculate_model_losses(skip_pixel_loss, model, imgs, imgs_pred,
                                                        boxes, boxes_pred, masks, masks_pred,
                                                        predicates, predicate_scores, l1_pixel_weight,
                                                        bbox_pred_loss_weight,
                                                        predicate_pred_loss_weight, mask_loss_weight)

            total_iou += jaccard(boxes_pred, boxes)
            total_boxes += boxes_pred.size(0)

            for loss_name, loss_val in losses.items():
                all_losses[loss_name].append(loss_val)
            num_samples += imgs.size(0)
            if num_samples >= num_val_samples:
                break

        samples = {}
        samples['gt_img'] = imgs

        model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
        samples['gt_box_gt_mask'] = model_out[0]

        model_out = model(objs, triples, obj_to_img, boxes_gt=boxes)
        samples['gt_box_pred_mask'] = model_out[0]

        model_out = model(objs, triples, obj_to_img)
        samples['pred_box_pred_mask'] = model_out[0]

        for k, v in samples.items():
            samples[k] = imagenet_deprocess_batch(v)

        mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
        avg_iou = total_iou / total_boxes

        masks_to_store = masks
        if masks_to_store is not None:
            masks_to_store = masks_to_store.data.cpu().clone()

        masks_pred_to_store = masks_pred
        if masks_pred_to_store is not None:
            masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

    batch_data = {
        'objs': objs.detach().cpu().clone(),
        'boxes_gt': boxes.detach().cpu().clone(),
        'masks_gt': masks_to_store,
        'triples': triples.detach().cpu().clone(),
        'obj_to_img': obj_to_img.detach().cpu().clone(),
        'triple_to_img': triple_to_img.detach().cpu().clone(),
        'boxes_pred': boxes_pred.detach().cpu().clone(),
        'masks_pred': masks_pred_to_store
    }
    out = [mean_losses, samples, batch_data, avg_iou]

    return tuple(out)


def calculate_model_losses(skip_pixel_loss, model, img, img_pred,
                           bbox, bbox_pred, masks, masks_pred,
                           predicates, predicate_scores, l1_pixel_weight, bbox_pred_loss_weight,
                           predicate_pred_loss_weight, mask_loss_weight):
    total_loss = torch.zeros(1).to(img)
    losses = {}

    l1_pixel_loss = F.l1_loss(img_pred, img)
    total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                          l1_pixel_weight)
    loss_bbox = F.mse_loss(bbox_pred, bbox)
    total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                          bbox_pred_loss_weight)

    if predicate_pred_loss_weight > 0:
        loss_predicate = F.cross_entropy(predicate_scores, predicates)
        total_loss = add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                              predicate_pred_loss_weight)

    if mask_loss_weight > 0 and masks is not None and masks_pred is not None:
        mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
        total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss', mask_loss_weight)
    return total_loss, losses
