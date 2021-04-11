import tensorflow as tf
import pandas as pd

from utils.models.frcnn import Faster_RCNN
from utils.label_generator import gt_generator, label_generator
from utils.Augmentation import *
from utils.models.train_step import frcnn_train_step
from utils.utils import df_resize, Anchor_Boxes, anchors_to_coordinates

if __name__ == '__main__':
    train_val_dir = 'data/'

    Ry = 0.4
    Rx = 0.4
    size = (432, 768)
    train = pd.read_csv('train.csv')
    valid = pd.read_csv('valid.csv')

    train = train.sample(frac=1).reset_index(drop=True)
    valid = valid.sample(frac=1).reset_index(drop=True)
    train = df_resize(train, Rx, Ry)
    valid = df_resize(valid, Rx, Ry)

    def traingtGenerator():
        Rx, Ry = 0.4, 0.4
        image_size = (1080, 1920, 3)
        size = [int(image_size[0] * Rx), int(image_size[1] * Ry)]
        iter_num = len(train)

        for i in range(iter_num):
            img = tf.io.read_file(train_val_dir + 'train/' + train['image'].iloc[i]) 
            img = tf.image.decode_jpeg(img, channels=3) 
            img = tf.image.resize(img, size) 
            img = img/255                         
            target = list(train.iloc[:,1:49].iloc[i,:])
            gt = gt_generator(target)
            cls_label, reg_label = label_generator(gt, anchor_boxes, out_boundaries_indxes)

            yield img, (cls_label, reg_label, gt)
        
        for i in range(iter_num):
            img = tf.io.read_file(train_val_dir + 'train/' + train['image'].iloc[i]) 
            img = tf.image.decode_jpeg(img, channels=3) 
            img = tf.image.resize(img, size) 
            img = img/255
            target = train.iloc[:,1:49].iloc[i,:] 
            img, target = left_right_flip(img, target)
            gt = gt_generator(target)
            cls_label, reg_label = label_generator(gt, anchor_boxes, out_boundaries_indxes)

            yield img, (cls_label, reg_label, gt)

        for i in range(iter_num):
            img = tf.io.read_file(train_val_dir + 'train/' + train['image'].iloc[i]) 
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, size)
            img = img/255
            target = train.iloc[:,1:49].iloc[i,:]
            img_list, target_list = shift_images(img, target)
            for shifted_img, shifted_target in zip(img_list, target_list):
                gt = gt_generator(shifted_target)
                cls_label, reg_label = label_generator(gt, anchor_boxes, out_boundaries_indxes)

                yield shifted_img, (cls_label, reg_label, gt)

    def valGenerator():
        Rx, Ry = 0.4, 0.4
        image_size = (1080, 1920, 3)
        size = [int(image_size[0] * Rx), int(image_size[1] * Ry)]

        for i in range(len(valid)):
            img = tf.io.read_file(train_val_dir + 'val/' + valid['image'].iloc[i]) 
            img = tf.image.decode_jpeg(img, channels=3) 
            img = tf.image.resize(img, size) 
            img = img/255     

            target = list(valid.iloc[:,1:49].iloc[i,:])
            gt = gt_generator(target)
            cls_label, reg_label = label_generator(gt, anchor_boxes, out_boundaries_indxes)         

            yield img, (cls_label, reg_label, gt)

    scales = [140, 160, 180, 210, 240]
    ratio = [(1/np.sqrt(3), np.sqrt(3)), (1/np.sqrt(2), np.sqrt(2)), (1, 1), (np.sqrt(2), 1/np.sqrt(2)), (np.sqrt(3), 1/np.sqrt(3))]
    anchor_boxes = Anchor_Boxes((432, 768, 3), scales, ratio)

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

    valid_dataset = tf.data.Dataset.from_generator(
        valGenerator,
        output_signature = (
                tf.TensorSpec(shape=(size[0], size[1], 3)),
                (
                    tf.TensorSpec(shape=(len(anchor_boxes))),
                    tf.TensorSpec(shape=(len(anchor_boxes),4)),
                    tf.TensorSpec(shape=(4))
                )
            )
    ).batch(batch_size).prefetch(16*4)    

    img_size=(432, 768, 3)
    anchor_boxes=anchor_boxes
    k=5*5
    n_sample=32
    backbone='resnet50'
    rpn_lambda=10**3
    pool_size=7

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
            valid_dataset=valid_dataset,
            train_stage=i,
            change_lr=False,
            rpn_lr=None,
            cls_lr=None,
            epochs=10
        )

    frcnn.save_weights("./model_weight/frcnn")