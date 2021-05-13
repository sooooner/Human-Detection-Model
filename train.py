import tensorflow as tf
import pandas as pd
import pymysql
import base64
import config

from utils.models.frcnn import Faster_RCNN
from utils.label_generator import gt_generator, label_generator
from utils.Augmentation import *
from utils.models.train_step import frcnn_train_step
from utils.utils import df_resize, Anchor_Boxes, anchors_to_coordinates

def main():
    Ry = 0.4
    Rx = 0.4
    size = (432, 768)

    conn = pymysql.connect(host='mysql', user='root', charset='utf8') 
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    sql = "USE dacon;" 
    cursor.execute(sql) 

    sql = "SELECT * FROM images LEFT JOIN keypoint on keypoint.id = images.image_id;" 
    cursor.execute(sql) 
    result = cursor.fetchall()
    train = pd.json_normalize(result)

    # SQL loader
    def traingtGenerator():
        for i in range(len(train)):
            img = base64.decodebytes(train.iloc[i]['image'])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [432, 768])/255

            target = list(train.iloc[0, 5:])
            gt = gt_generator(target)
            cls_label, reg_label = label_generator(gt, anchor_boxes, out_boundaries_indxes)
            
            yield img, (cls_label, reg_label, gt)
        
        for i in range(len(train)):
            img = base64.decodebytes(train.iloc[i]['image'])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [432, 768])/255

            target = list(train.iloc[0, 5:])
            img, target = left_right_flip(img, target)
            gt = gt_generator(target)
            cls_label, reg_label = label_generator(gt, anchor_boxes, out_boundaries_indxes)

            yield img, (cls_label, reg_label, gt)

        for i in range(len(train)):
            img = base64.decodebytes(train.iloc[i]['image'])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [432, 768])/255

            target = list(train.iloc[0, 5:])
            img_list, target_list = shift_images(img, target)
            for shifted_img, shifted_target in zip(img_list, target_list):
                gt = gt_generator(shifted_target)
                cls_label, reg_label = label_generator(gt, anchor_boxes, out_boundaries_indxes)

                yield shifted_img, (cls_label, reg_label, gt)

        for i in range(len(train)):
            img = base64.decodebytes(train.iloc[i]['image'])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [432, 768])/255

            target = list(train.iloc[0, 5:])
            noisy_img = add_noise(img)
            gt = gt_generator(target)
            cls_label, reg_label = label_generator(gt, anchor_boxes, out_boundaries_indxes)

            yield noisy_img, (cls_label, reg_label, gt)


    scales = config.SCALES
    ratio = config.RATIO
    anchor_boxes = Anchor_Boxes(config.IMG_SIZE, scales, ratio)

    bboxes = anchors_to_coordinates(anchor_boxes)
    out_boundaries_indxes = (np.where(bboxes[:, 0] < 0) or np.where(bboxes[:, 2] < 0) or np.where(bboxes[:, 1] > 768) or np.where(bboxes[:, 3] > 432))[0]

    batch_size = 16
    train_dataset = tf.data.Dataset.from_generator(
        traingtGenerator,
        output_signature = (
                tf.TensorSpec(shape=(size[0], size[1], 3)),
                (
                    tf.TensorSpec(shape=(len(anchor_boxes))),
                    tf.TensorSpec(shape=(len(anchor_boxes),4)),
                    tf.TensorSpec(shape=(4))
                )
            )
    ).batch(batch_size).prefetch(16*4)


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

    frcnn.compile(
        rpn_optimizer = tf.keras.optimizers.Adam(lr=0.01),
        classifier_optimizer = tf.keras.optimizers.Adam(lr=0.001)
    )

    for i in [1, 2, 3, 4]:
        frcnn = frcnn_train_step(
            model=frcnn, 
            train_dataset=train_dataset, 
            train_stage=i,
            change_lr=False,
            rpn_lr=None,
            cls_lr=None,
            epochs=10
        )

    frcnn.save_weights("./saved_model/1")

if __name__ == '__main__':
    main()