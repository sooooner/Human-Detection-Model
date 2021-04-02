import tensorflow as tf
import numpy as np

def IoU_np(box1, anchor_boxes):
    '''
    numpy ver
    inputs
    box1 : ground truth box
    anchor_boxes : anchor boxes
    '''
    broadcast = len(anchor_boxes)
    
    box1_area = box1[2] * box1[3]
    box2_area = anchor_boxes[:,2] * anchor_boxes[:,3]
    
    x1 = np.max([np.broadcast_to(box1[0] - box1[2]/2, broadcast), anchor_boxes[:, 0] - anchor_boxes[:, 2]/2], axis=0)
    x2 = np.min([np.broadcast_to(box1[0] + box1[2]/2, broadcast), anchor_boxes[:, 0] + anchor_boxes[:, 2]/2], axis=0)
    
    y1 = np.max([np.broadcast_to(box1[1] - box1[3]/2, broadcast), anchor_boxes[:, 1] - anchor_boxes[:, 3]/2], axis=0)
    y2 = np.min([np.broadcast_to(box1[1] + box1[3]/2, broadcast), anchor_boxes[:, 1] + anchor_boxes[:, 3]/2], axis=0)
    
    h = np.max([np.broadcast_to(0.0, broadcast), y2 - y1 + 1], axis=0)
    w = np.max([np.broadcast_to(0.0, broadcast), x2 - x1 + 1], axis=0)
    
    intersect = h * w
    union = np.broadcast_to(box1_area, broadcast) + box2_area - intersect
    return intersect / union 


def IoU_tensor(rois, gts):
    '''
    tensor ver
    inputs
    box1 : ground truth boxes
    anchor_boxes : anchor boxes
    '''
    box1_area = tf.cast(rois[:, :, 2] * rois[:, :, 2], tf.float64)
    box2_area = tf.cast(gts[:,2] * gts[:,3], tf.float64)
    
    x1 = tf.maximum(tf.cast(rois[:, :, 0] - rois[:, :, 2]/2, tf.float64), tf.cast(tf.expand_dims(gts[:, 0] - gts[:, 2]/2, -1), tf.float64))
    x2 = tf.minimum(tf.cast(rois[:, :, 0] + rois[:, :, 2]/2, tf.float64), tf.cast(tf.expand_dims(gts[:, 0] + gts[:, 2]/2, -1), tf.float64))
    y1 = tf.maximum(tf.cast(rois[:, :, 1] - rois[:, :, 3]/2, tf.float64), tf.cast(tf.expand_dims(gts[:, 1] - gts[:, 3]/2, -1), tf.float64))
    y2 = tf.minimum(tf.cast(rois[:, :, 1] + rois[:, :, 3]/2, tf.float64), tf.cast(tf.expand_dims(gts[:, 1] + gts[:, 3]/2, -1), tf.float64))
    
    h = tf.maximum(tf.constant(0.0, dtype=tf.float64), y2 - y1)
    w = tf.maximum(tf.constant(0.0, dtype=tf.float64), x2 - x1)
    
    intersect = tf.math.multiply(h, w)
    union = tf.subtract(tf.add(box1_area, tf.expand_dims(box2_area, -1)), intersect)
    return tf.divide(intersect, union)