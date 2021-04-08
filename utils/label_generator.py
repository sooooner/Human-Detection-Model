import numpy as np
import tensorflow as tf
import pandas as pd

from utils.intersection_over_union import *

def ground_truth_generator(df):
    df['ratio'] = df.apply(lambda x: 'w' if x['x_max'] - x['x_min'] > x['y_max'] - x['y_min'] else 'h', axis=1)
    df['x_min'] = df.apply(lambda x: x['x_min'] - (x['x_max'] - x['x_min'])*.1 if x['ratio'] == 'h' else x['x_min'], axis=1)
    df['x_max'] = df.apply(lambda x: x['x_max'] + (x['x_max'] - x['x_min'])*.1 if x['ratio'] == 'h' else x['x_max'], axis=1)
    df['y_min'] = df.apply(lambda x: x['y_min'] - (x['y_max'] - x['y_min'])*.1 if x['ratio'] == 'w' else x['y_min'], axis=1)
    df['y_max'] = df.apply(lambda x: x['y_max'] + (x['y_max'] - x['y_min'])*.1 if x['ratio'] == 'w' else x['y_max'], axis=1)

    ground_truth = df.iloc[:,:1].copy() 
    ground_truth['x_min'] = df['x_min'] - (df['x_max'] - df['x_min'])*.05
    ground_truth['x_max'] = df['x_max'] + (df['x_max'] - df['x_min'])*.05
    ground_truth['y_min'] = df['y_min'] - (df['y_max'] - df['y_min'])*.05
    ground_truth['y_max'] = df['y_max'] + (df['y_max'] - df['y_min'])*.05

    ground_truth['w'] = ground_truth['x_max'] - ground_truth['x_min']
    ground_truth['h'] = ground_truth['y_max'] - ground_truth['y_min']
    ground_truth['x'] = ground_truth['w']/2 + ground_truth['x_min']
    ground_truth['y'] = ground_truth['h']/2 + ground_truth['y_min']
    return ground_truth

def gt_generator(target):
    x_min = np.min(target[::2])
    x_max = np.max(target[::2])
    y_min = np.min(target[1::2])
    y_max = np.max(target[1::2])
    ratio = 'w' if x_max - x_min > y_max - y_min else 'h'

    x_min = x_min - (x_max - x_min)*.1 if ratio == 'h' else x_min
    x_max = x_max + (x_max - x_min)*.1 if ratio == 'h' else x_max
    y_min = y_min - (y_max - y_min)*.1 if ratio == 'w' else y_min
    y_max = y_max + (y_max - y_min)*.1 if ratio == 'w' else y_max

    ground_truth_x_min = x_min - (x_max - x_min)*.05
    ground_truth_x_max = x_max + (x_max - x_min)*.05
    ground_truth_y_min = y_min - (y_max - y_min)*.05
    ground_truth_y_max = y_max + (y_max - y_min)*.05

    ground_truth_w = ground_truth_x_max - ground_truth_x_min
    ground_truth_h = ground_truth_y_max - ground_truth_y_min
    ground_truth_x = ground_truth_w/2 + ground_truth_x_min
    ground_truth_y = ground_truth_h/2 + ground_truth_y_min
    return [ground_truth_x, ground_truth_y, ground_truth_w, ground_truth_h], [ground_truth_x_min, ground_truth_y_min]

def label_generator(GT, anchor_boxes, out_boundaries_indxes):
    cls_label = - np.ones(shape=(anchor_boxes.shape[0]))
    pos_iou_threshold = 0.7
    neg_iou_threshold = 0.3
    n_sample = 32
    pos_ratio = 0.5
    n_pos = int(pos_ratio * n_sample)
    
    ious = np.apply_along_axis(IoU_np, 0, GT, anchor_boxes=anchor_boxes)
    cls_label[ious >= pos_iou_threshold] = 1
    cls_label[ious < neg_iou_threshold] = 0
    cls_label[np.argmax(ious)] = 1
    cls_label[out_boundaries_indxes] = -1

    pos_index = np.where(cls_label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(
            pos_index,
            size = (len(pos_index) - n_pos),
            replace=False
        )
        cls_label[disable_index] = -1

    n_neg = n_sample - np.sum(cls_label == 1)
    neg_index = np.where(cls_label == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(
            neg_index, 
            size = (len(neg_index) - n_neg),             
            replace = False
        )
        cls_label[disable_index] = -1

    reg_label = anchor_boxes * np.broadcast_to(tf.cast(cls_label > 0, tf.int32), (4, len(cls_label))).T
    indices = np.where(reg_label != 0)[0][::4]
    x, y, w, h = GT[0], GT[1], GT[2], GT[3]

    tx = (x - reg_label[indices][:, 0]) / (reg_label[indices][:, 2])
    ty = (y - reg_label[indices][:, 1]) / (reg_label[indices][:, 3])
    tw = np.log(w / reg_label[indices][:, 2]) 
    th = np.log(h / reg_label[indices][:, 3]) 
    reg_label[indices] = np.stack([tx, ty, tw, th]).T

    return cls_label, reg_label
    

def classifier_label_generator(nms, gts, valid=False):
    ious = get_iou(nms, gts)
    pos_order = tf.argsort(ious * tf.cast(tf.math.greater_equal(ious, 0.5), tf.float64), direction='DESCENDING', axis=1)[:, :32]
    
    if valid:
        neg_order = tf.argsort(ious * tf.cast(tf.math.less(ious, 0.5), tf.float64), direction='DESCENDING', axis=1)[:, :96]
    else:
        neg_cnt = tf.reduce_min(tf.math.count_nonzero(ious * tf.cast(tf.math.less(ious, 0.5), tf.float64), axis=1))
        neg_order = tf.argsort(ious * tf.cast(tf.math.less(ious, 0.5), tf.float64), direction='DESCENDING', axis=1)[:, :neg_cnt]
        indices = tf.range(start=0, limit=neg_cnt, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        neg_order = tf.gather(neg_order, shuffled_indices, axis=1)[:, :96]

    cls_labels = tf.concat([tf.ones_like(pos_order), tf.zeros_like(neg_order)], axis=1)
    label_order = tf.concat([pos_order, neg_order], axis=1)
    P_boxes = tf.gather(nms, label_order, batch_dims=1)
    
    tx = (gts[:, :1] - P_boxes[:, :, 0]) / P_boxes[:, :, 2]
    ty = (gts[:, 1:2] - P_boxes[:, :, 1]) / P_boxes[:, :, 3]
    tw = tf.math.log(gts[:, 2:3] / P_boxes[:, :, 2]) 
    th = tf.math.log(gts[:, 3:] / P_boxes[:, :, 3]) 
    tf.stack([tx, ty, tw, th], axis=-1)
    box_labels = tf.stack([tx, ty, tw, th], axis=-1)
    return box_labels, cls_labels, P_boxes