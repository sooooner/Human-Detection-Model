import tensorflow as tf
import pandas as pd
import numpy as np

from utils.models.frcnn import Faster_RCNN
from utils.utils import Anchor_Boxes

if __name__ == '__main__':
    batch_size = 16
    img_size=(432, 768, 3)
    scales = [140, 160, 180, 210, 240]
    ratio = [(1/np.sqrt(3), np.sqrt(3)), (1/np.sqrt(2), np.sqrt(2)), (1, 1), (np.sqrt(2), 1/np.sqrt(2)), (np.sqrt(3), 1/np.sqrt(3))]
    anchor_boxes=Anchor_Boxes(img_size, scales, ratio)
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

    frcnn.load_weights("./model_weight/frcnn")
    test = pd.read_csv('res/sample_submission.csv')

    def testGenerator():
        for i in range(len(test)):
            img = tf.io.read_file('res/test_imgs/' + test['image'].iloc[i]) 
            img = tf.image.decode_jpeg(img, channels=3) 
            img = tf.image.resize(img, [432, 768]) 
            img = img/255                         

            yield img

    test_dataset = tf.data.Dataset.from_generator(testGenerator, (tf.float32), (tf.TensorShape([432, 768,3]))).batch(batch_size).prefetch(16*4)

    for x_test in test_dataset:
        scores, rps, feature_map = frcnn.rpn(x_test)
        rps = frcnn.rpn.inverse_bbox_regression(rps)
        candidate_area, scores = frcnn.get_candidate((scores, rps, frcnn.n_test_pre_nms))
        nms = frcnn.get_nms((candidate_area, scores, frcnn.n_test_post_nms))
        rois = frcnn.roipool((feature_map, nms))
        cls, bbox_reg, nms = frcnn.classifier((rois, nms))
        predict = frcnn.classifier.inverse_bbox_regression(bbox_reg, nms)