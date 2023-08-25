# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    total_iou = 0.0
    num_batches = len(data_loader)
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)   #dict_keys(['pred_logits': [4, 100, 255], 'pred_boxes': [4, 100, 4], 'aux_outputs']) 
        loss_dict = criterion(outputs, targets)
        label_values = torch.cat([item["boxes"] for item in targets])
        mean_iou = calculate_miou(label_values, outputs['pred_boxes'].squeeze())
        total_iou += mean_iou
        
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # if coco_evaluator is not None:
        #     coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name

        #     panoptic_evaluator.update(res_pano)

    mean_iou = total_iou / num_batches
    print("Mean IoU for the entire dataset:", mean_iou)
    return mean_iou, None

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # if panoptic_evaluator is not None:
    #     panoptic_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in postprocessors.keys():
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in postprocessors.keys():
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]
    # return stats, coco_evaluator

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two rectangles.

    Parameters:
    box1 (tuple): (x1, y1, x2, y2) coordinates of the first rectangle
    box2 (tuple): (x1, y1, x2, y2) coordinates of the second rectangle

    Returns:
    float: IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    intersection_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_miou(batch_boxes_true, batch_boxes_pred):
    """
    Calculate the mean Intersection over Union (mIoU) for a batch of data.

    Parameters:
    batch_boxes_true (list of tuples): List of true box coordinates for each data
    batch_boxes_pred (list of tuples): List of predicted box coordinates for each data

    Returns:
    float: Mean IoU value
    """
    total_iou = 0.0
    num_samples = len(batch_boxes_true)
    for i in range(num_samples):
        iou = calculate_iou(batch_boxes_true[i], batch_boxes_pred[i])
        total_iou += iou

    mean_iou = total_iou / num_samples
    return mean_iou
