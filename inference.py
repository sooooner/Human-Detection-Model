import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import config
import cv2

from utils.models.frcnn import Faster_RCNN
from utils.utils import Anchor_Boxes

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True, type=str, help="Data path")
args = ap.parse_args()

def main(img):
    batch_size = 16
    scales = config.SCALES
    ratio = config.RATIO
    anchor_boxes = Anchor_Boxes(config.IMG_SIZE, scales, ratio)
    img_size = config.IMG_SIZE
    anchor_boxes = anchor_boxes
    k = config.K
    n_sample = config.N_SAMPLE
    backbone = config.BACKBONE
    rpn_lambda = config.RPN_LAMBDA
    pool_size = config.POOL_SIZE

    frcnn = Faster_RCNN(
        img_size=img_size, 
        anchor_boxes=anchor_boxes, 
        k=k, 
        n_sample=n_sample, 
        backbone=backbone,
        rpn_lambda=rpn_lambda, 
        pool_size=pool_size
    )

    frcnn.load_weights("./model_weight/frcnn")

    cls, predict = frcnn(tf.expand_dims(img, 0))
    selected_indices = tf.image.non_max_suppression(predict[0], tf.squeeze(tf.cast(cls[0], tf.float32)), max_output_size=5, score_threshold=.5)
    anchors = tf.gather(predict[0], selected_indices)
    anchor = np.mean(anchors, axis=0)
    
    x1 = int(anchor[0] - anchor[2]/2)
    x2 = int(anchor[0] + anchor[2]/2)
    y1 = int(anchor[1] - anchor[3]/2)
    y2 = int(anchor[1] + anchor[3]/2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    fig, ax = plt.subplots(dpi=200)
    ax.imshow(img)
    ax.axis('off')
    plt.show()

if __name__ == '__main__':
    main(args.path)

    