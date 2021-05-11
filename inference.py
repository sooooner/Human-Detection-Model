import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import config
import cv2

from utils.models.frcnn import Faster_RCNN
from utils.utils import Anchor_Boxes, anchor_to_coordinate

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True, type=str, help="Image data path")
args = ap.parse_args()

def main(img_path):
    img = tf.io.read_file(img_path) 
    img_ = tf.image.decode_jpeg(img, channels=3)
    img = tf.expand_dims(tf.image.resize(img_, [432, 768])/255, 0)
    img_ = img_.numpy()

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

    frcnn.load_weights("./model_weight/step4").expect_partial()

    cls, predict = frcnn(img)
    max_output_size = 5

    scores_order = tf.argsort(cls[0], direction='DESCENDING', axis=0)
    boxes = tf.squeeze(tf.gather(predict[0], scores_order))
    boxes = boxes[boxes[:, 2] > 16]
    boxes = boxes[boxes[:, 3] > 16][:max_output_size]
        
    anchor = anchor_to_coordinate(np.mean(boxes, axis=0))
    cv2.rectangle(
        img_, 
        (int(anchor[0]*2.5), int(anchor[2]*2.5)), (int(anchor[1]*2.5), int(anchor[3]*2.5)), 
        (255, 0, 0), 
        thickness=2
    )

    plt.imshow(img_)
    plt.axis('off')
    plt.savefig('fig1.png', dpi=200)

if __name__ == '__main__':
    # python inference.py --path=./res/train_imgs/001-1-1-01-Z17_A-0000001.jpg
    main(args.path)

    