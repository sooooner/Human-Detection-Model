import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def min_max_cal(df):
    df['x_min'] = df.iloc[:, 1:49:2].apply(lambda x: int(min(x)), axis=1)
    df['x_max'] = df.iloc[:, 1:49:2].apply(lambda x: int(max(x)), axis=1)
    df['y_min'] = df.iloc[:, 2:49:2].apply(lambda x: int(min(x)), axis=1)
    df['y_max'] = df.iloc[:, 2:49:2].apply(lambda x: int(max(x)), axis=1)
    return df

def df_resize(df, Rx, Ry):
    df_new = df.copy()
    df_new.iloc[:, 1:49:2] = df_new.iloc[:, 1:49:2] * Rx
    df_new.iloc[:, 2:49:2] = df_new.iloc[:, 2:49:2] * Ry
    df_new = min_max_cal(df_new)
    return df_new
    
def anchor_box_generator(x, y, scales, ratio):
    anchor_boxes = []
    for scale in scales:
        for w, h  in ratio:
            w *= scale
            h *= scale     
            anchor_boxes.append([x, y, w, h])
    return anchor_boxes

def Anchor_Boxes(img_shape, scales, ratio, model='vgg'):
    '''
    input
    img_shape : image shape
    output 
    numpy array shape (w * h * k, 4)
    '''
    if model == 'vgg':
        Ratio = 2**4
        
    w=img_shape[1]//Ratio
    h=img_shape[0]//Ratio
    
    anchor_boxes = []
    for x in range(img_shape[1]//w//2, img_shape[1], img_shape[1]//w):
        for y in range(img_shape[0]//h//2, img_shape[0], img_shape[0]//h):
            anchor_boxes.append(anchor_box_generator(x, y, scales, ratio))
    return np.array(anchor_boxes).reshape(-1, 4)

def anchors_to_coordinates(bboxes):
    x1 = bboxes[:, 0] - bboxes[:, 2]/2
    x2 = bboxes[:, 0] + bboxes[:, 2]/2
    y1 = bboxes[:, 1] - bboxes[:, 3]/2
    y2 = bboxes[:, 1] + bboxes[:, 3]/2
    return np.stack([x1, x2, y1, y2], axis=-1)

def coordinates_to_anxhors(boxes):
    w = boxes[:, :, 1] - boxes[:, :, 0]
    h = boxes[:, :, 3] - boxes[:, :, 2]
    x = boxes[:, :, 0] + w/2
    y = boxes[:, :, 2] + h/2
    return tf.stack([x, y, w, h], axis=-1)

def anchor_to_coordinate(box):    
    x1 = box[0] - box[2]/2
    x2 = box[0] + box[2]/2
    y1 = box[1] - box[3]/2
    y2 = box[1] + box[3]/2
    return (x1, x2, y1, y2)

def anchors_to_coordinates_clip(boxes, size=(432, 768)):    
    x1 = boxes[:, :, 0] - boxes[:, :, 2]/2
    x2 = boxes[:, :, 0] + boxes[:, :, 2]/2
    y1 = boxes[:, :, 1] - boxes[:, :, 3]/2
    y2 = boxes[:, :, 1] + boxes[:, :, 3]/2
    
    x1 = tf.clip_by_value(x1, 0, size[1])
    x2 = tf.clip_by_value(x2, 0, size[1])
    y1 = tf.clip_by_value(y1, 0, size[0])
    y2 = tf.clip_by_value(y2, 0, size[0])
    return tf.stack([x1, x2, y1, y2], axis=-1)


def get_rois(scores, rps, anchor_boxes):
    gx = anchor_boxes[:, 0] + anchor_boxes[:, 2] * rps[:,:,0]
    gy = anchor_boxes[:, 1] + anchor_boxes[:, 3] * rps[:,:,1]
    gw = anchor_boxes[:, 2] * tf.exp(rps[:, :, 2])
    gh = anchor_boxes[:, 3] * tf.exp(rps[:, :, 3])
    rois = tf.stack([gx, gy, gw, gh], axis=-1)
    rois = anchors_to_coordinates_clip(rois)
    
    n_train_pre_nms = 3000
    orders = tf.argsort(scores, axis=1)[:, ::-1, :]
    orders = orders[:, :n_train_pre_nms]
    rois = tf.reshape(tf.gather(rois, orders, batch_dims=1), (-1, n_train_pre_nms, 4))
    scores = tf.squeeze(tf.gather(scores, orders, batch_dims=1))
    rois = coordinates_to_anxhors(rois)
    return rois, scores


def bbox_nms(img, candidate_area, scores, max_output_size=5, ground_truth_row=False):
    colors = {k: tuple(map(float, np.random.randint(0, 255, 3)/255)) for k in range(max_output_size)}
    img_ = img.copy()

    if ground_truth_row is not False:
        x1 = int(ground_truth_row['x_min'])
        x2 = int(ground_truth_row['x_max'])
        y1 = int(ground_truth_row['y_min'])
        y2 = int(ground_truth_row['y_max'])
        cv2.rectangle(img_, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

    selected_indices = tf.image.non_max_suppression(candidate_area, tf.squeeze(tf.cast(scores , tf.float32)), max_output_size=max_output_size, score_threshold=1e-12, iou_threshold=.7)
    anchors = tf.gather(candidate_area, selected_indices)

    for i, anchor in enumerate(anchors):
        anchor = anchor_to_coordinate(anchor.numpy())
        cv2.rectangle(
            img_, 
            (int(anchor[0]), int(anchor[2])), (int(anchor[1]), int(anchor[3])), 
            colors.get(i), 
            thickness=1
        )

    fig, ax = plt.subplots(dpi=200)
    ax.imshow(img_)
    ax.axis('off')
    plt.show()