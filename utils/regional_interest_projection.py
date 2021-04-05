import tensorflow as tf

def anchor_to_anchor(boxes):    
    x1 = boxes[:, :, 0] - boxes[:, :, 2]/2
    y1 = boxes[:, :, 1] - boxes[:, :, 3]/2
    return tf.stack([y1, x1, boxes[:, :, 3], boxes[:, :, 2]], axis=-1)

def projection(nmses, feature_map, batch_size, n_sample):
    roi_projections_tensor = []
    for i in range(batch_size):
        roi_projections = []
        for j in range(n_sample):
            y, x, h, w = nmses[i, j]
            roi_projection = tf.image.crop_to_bounding_box(feature_map[i], y, x, h, w)
            roi_projection = tf.image.resize(roi_projection, size=(7, 13))
            roi_projections.append(roi_projection)
        roi_projections_tensor.append(tf.stack(roi_projections, axis=0))
    return tf.stack(roi_projections_tensor, axis=0)

def box_regression(boxes, gt):
    boxes = tf.cast(boxes, tf.float64)
    gt = tf.cast(gt, tf.float64)

    x = tf.reshape(gt[:, 0], (-1, 1))
    y = tf.reshape(gt[:, 1], (-1, 1))
    w = tf.reshape(gt[:, 2], (-1, 1))
    h = tf.reshape(gt[:, 3], (-1, 1))

    tx = (x - boxes[:, :, 0]) / (boxes[:, :, 2])
    ty = (y - boxes[:, :, 1]) / (boxes[:, :, 3])
    tw = tf.math.log(w / boxes[:, :, 2]) 
    th = tf.math.log(h / boxes[:, :, 3]) 
    return tf.stack([tx, ty, tw, th], -1)

def NMS(rois, scores, feature_map):
    n_sample = 32
    pos_score_thresh = 0.5 
    
    batch_size = 16
    if rois.shape[0]:
        batch_size = rois.shape[0]

    nmses = []
    for i in range(batch_size):
        selected_indices = tf.image.non_max_suppression(
            rois[i], 
            tf.cast(scores[i], tf.float32), 
            max_output_size=100, 
            iou_threshold=pos_score_thresh
        )
        selected_anchors = tf.gather(rois[i], selected_indices)
        selected_scores = tf.gather(scores[i], selected_indices)

        pos_indices = tf.squeeze(tf.where(selected_scores >= pos_score_thresh), -1)
        pos_anchors = tf.gather(selected_anchors, pos_indices) 

        neg_indices =tf.squeeze(tf.where(selected_scores < pos_score_thresh), -1)
        neg_anchors = tf.gather(selected_anchors, neg_indices)
        
        labels = tf.reshape(tf.concat([tf.ones_like(pos_anchors)[:,0], tf.zeros_like(neg_anchors)[:,0]], axis=0)[:n_sample], (-1, 1))
        nms = tf.concat([pos_anchors, neg_anchors], axis=0)[:n_sample]
        nms = tf.concat([nms, labels], axis=-1)
        nmses.append(nms)

    nmses = tf.stack(nmses, axis=-1)
    cls_labels = tf.expand_dims(nmses[:, :, -1], -1)
    nmses = nmses[:, :, :4]

    nmss = tf.cast(anchor_to_anchor(nmses)//2**4, tf.int32)
    roi_projections_tensor = projection(nmss, feature_map, batch_size, n_sample)

    return roi_projections_tensor, nmses, cls_labels


