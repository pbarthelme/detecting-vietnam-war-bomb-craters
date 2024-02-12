"""Evaluation functions

A collection of functions that are used for model evaluation.

This script requires the numpy, pandas, scipy, scikit-learn, scikit-image and pyyaml packages 
to be installed within the Python environment you are running this script in.

This file contains the following functions:

    * create_dir - create a new directory
    * load_config - load a YAML configuration file
    * apply_min_max_scaling - apply min-max scaling to a numpy array
    * load_data - load data from a h5py file
    * rect_from_bound - return rectangle coordinates from a list of bounds
"""

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from skimage import morphology
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def combine_classes(x, class_ids):
    idx = np.isin(x, class_ids)
    x[~idx] = 0
    x[idx] = 1
    return x


def evaluate_pixel_accuracy(pred, y, crater_ids=None):
    pred_class = np.argmax(pred, axis=-1).flatten()
    true_class = np.argmax(y, axis=-1).flatten()
    res = precision_recall_fscore_support(true_class, pred_class)
    df_res = pd.DataFrame(
        np.round(res, 3),
        index=["Precision", "Recall", "F1-Score", "N"]
    )

    if crater_ids:
        pred_class = combine_classes(pred_class, class_ids=crater_ids)
        true_class = combine_classes(true_class, class_ids=crater_ids)
        res = precision_recall_fscore_support(true_class, pred_class)
        df_res["craters_combined"] = np.round([metric[1] for metric in res], 3)

    return df_res


def filter_by_size(x, min_crater_area, filled_id):
    x_mask = x.copy()
    x_mask = morphology.remove_small_holes(
        x_mask > 0, area_threshold=min_crater_area)
    x_mask = morphology.remove_small_objects(x_mask, min_size=min_crater_area)
    x[~x_mask] = 0
    # assign unused id filled in holes as assigning one of the crater class ids would affected
    # pixel counts for later determination of crater class based on max number of pixels in crater
    x[(x_mask) & (x == 0)] = filled_id

    return x


def calc_iou(true, pred, true_class, pred_class, n_classes):
    # Count objects
    true_objects = len(np.unique(true))
    pred_objects = len(np.unique(pred))

    # compute class with largest number of pixels for each predicted and labelled crater
    # n_classes + 3: +1 background class, +1 for filled_in class and +1 for the way bins are created
    h_pred_class = np.histogram2d(
        pred.flatten(),
        pred_class.flatten(),
        bins=(pred_objects, range(n_classes+3)))

    # get majority class after removing last column to avoid "neutral" class being the majority
    pred_majority = np.argmax(h_pred_class[0][:, :-1], axis=-1)

    # compute label class
    h_true_class = np.histogram2d(
        true.flatten(), true_class.flatten(), bins=(true_objects, range(n_classes+2)))
    true_majority = np.argmax(h_true_class[0], axis=-1)

    # Compute intersection between true and preds
    h = np.histogram2d(true.flatten(), pred.flatten(),
                       bins=(true_objects, pred_objects))
    intersection = h[0]

    # Area of objects
    area_true = np.histogram(true, bins=true_objects)[0]
    area_pred = np.histogram(pred, bins=pred_objects)[0]

    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    pred_majority = pred_majority[1:]
    true_majority = true_majority[1:]

    # Compute Intersection over Union
    union[union == 0] = 1e-9
    iou = intersection/union

    return iou, true_majority, pred_majority


def calc_measures(threshold, iou, gt_class, pred_class, img_id=0, match_all=False):
    # unique assignment only possible with IOU threshold > 0.5, otherwise perform hungarian matching
    # between predicted and gt craters, maximizing overall IOU
    if match_all:
        # all predicted and gt craters in the same image tile are matched even if
        # they don't overlap, matching here does not account for crater classes
        # but can help to compare overall crater counts in each image tile
        match_idxes = linear_sum_assignment(iou, maximize=True)
        matches = np.zeros(iou.shape)
        matches[match_idxes] = 1

    elif threshold <= 0.5:
        iou[iou <= threshold] = 0
        match_idxes = linear_sum_assignment(iou, maximize=True)
        matches = np.zeros(iou.shape)
        matches[match_idxes] = 1
        # remove all matches that have IOU of 0 or were set to 0 as they
        # are smaller than the threshold
        matches[iou == 0] = 0

    else:
        matches = iou > threshold

    false_positives = np.sum(matches, axis=0) == 0
    false_negatives = np.sum(matches, axis=1) == 0

    # true positive craters with classes
    tp_preds_ids = np.where(matches)[1]
    tp_gts_ids = np.where(matches)[0]
    tp_preds = pred_class[tp_preds_ids]
    tp_gts = gt_class[tp_gts_ids]

    # add 1 to all ids as we removed the background class in the IOU table
    # reducing the dimension by 1
    tp_classes = [(img_id, "tp", id_pred+1, id_gt+1, pred, gt) for id_pred,
                  id_gt, pred, gt in zip(tp_preds_ids, tp_gts_ids, tp_preds, tp_gts)]

    # false positive craters with classes
    fp_ids = np.where(false_positives)[0]
    fp_preds = pred_class[false_positives]
    fp_classes = [(img_id, "fp", id_pred+1, 0, pred, 0)
                  for id_pred, pred in zip(fp_ids, fp_preds)]

    # false negative craters with classes
    fn_ids = np.where(false_negatives)[0]
    fn_gts = gt_class[false_negatives]
    fn_classes = [(img_id, "fn", 0, id_gt+1, 0, gt)
                  for id_gt, gt in zip(fn_ids, fn_gts)]

    return [*tp_classes, *fp_classes, *fn_classes]


def process_crater_pred_individual(pred, y, min_crater_area, n_classes, crater_ids,
                                   threshold=0.5, match_all=False):

    pred_class = np.argmax(pred, axis=-1)
    true_class = y

    # set all non crater classes as background
    pred_class[~np.isin(pred_class, crater_ids)] = 0
    true_class[~np.isin(true_class, crater_ids)] = 0

    filled_id = np.max(crater_ids) + 1

    pred_class = filter_by_size(
        pred_class, min_crater_area=min_crater_area, filled_id=filled_id)

    pred_crater = morphology.label(
        pred_class > 0, background=0, connectivity=1)
    true_crater = morphology.label(
        true_class > 0, background=0, connectivity=1)

    iou, true_class, pred_class = calc_iou(
        true_crater, pred_crater, true_class, pred_class, n_classes=n_classes)
    craters = calc_measures(threshold, iou, true_class,
                          pred_class, match_all=match_all)

    return pred_class, pred_crater, true_crater, craters


def process_crater_pred(pred, y, min_crater_area, n_classes, crater_ids,
                        threshold=0.5, match_all=False):

    pred_class_all = np.argmax(pred, axis=-1)
    true_class_all = np.argmax(y, axis=-1)

    # set all non crater classes as background
    pred_class_all[~np.isin(pred_class_all, crater_ids)] = 0
    true_class_all[~np.isin(true_class_all, crater_ids)] = 0

    # process each tile
    craters = list()
    filled_id = np.max(crater_ids) + 1

    i = 0
    for pred_class, true_class in zip(pred_class_all, true_class_all):
        pred_class = filter_by_size(
            pred_class, min_crater_area=min_crater_area, filled_id=filled_id)

        pred_crater = morphology.label(
            pred_class > 0, background=0, connectivity=1)
        true_crater = morphology.label(
            true_class > 0, background=0, connectivity=1)

        iou, true_class, pred_class = calc_iou(
            true_crater, pred_crater, true_class, pred_class, n_classes=n_classes)
        craters += calc_measures(threshold, iou, true_class,
                               pred_class, img_id=i, match_all=match_all)
        i += 1

    # convert to pandas dataframe
    df = pd.DataFrame(craters, columns=[
                      "image", "status", "pred_id", "label_id", "pred", "true"])
    df["pred_crater"] = df["pred"] > 0
    df["true_crater"] = df["true"] > 0

    return df


def evaluate_crater_accuracy(pred, y, min_crater_area, crater_ids, crater_classes,
                             threshold=0.5, match_all=False, plot_cm=True):
    df = process_crater_pred(
        pred=pred,
        y=y,
        min_crater_area=min_crater_area,
        n_classes=len(crater_classes),
        crater_ids=crater_ids,
        threshold=threshold,
        match_all=match_all
    )

    res = precision_recall_fscore_support(df["true"], df["pred"])
    df_res = pd.DataFrame(
        np.round(res, 3),
        index=["Precision", "Recall", "F1-Score", "N"]
    )

    res = precision_recall_fscore_support(df["true_crater"], df["pred_crater"])
    df_res["craters_combined"] = np.round([metric[1] for metric in res], 3)

    cm = confusion_matrix(df["true"], df["pred"])

    if plot_cm:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

    return df_res, cm
