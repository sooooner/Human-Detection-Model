import tensorflow as tf
from utils.models.rpn import RPN
from utils.models.classifier import Classifier
from utils.models.layers import *

class Faster_RCNN(tf.keras.models.Model):
    def __init__(self, img_size, anchor_boxes, k, n_sample, backbone, rpn_lambda, pool_size, **kwargs):
        super(Faster_RCNN, self).__init__(*kwargs)
        self.img_size = img_size
        self.anchor_boxes = anchor_boxes
        self.k = k
        self.n_sample = n_sample
        self.backbone = backbone
        self.rpn_lambda = rpn_lambda
        self.pool_size = pool_size
        self.n_train_pre_nms = 6000
        self.n_train_post_nms = 2000
        self.n_test_pre_nms = 6000
        self.n_test_post_nms = 300
        self.iou_threshold = 0.7

        self.rpn = RPN(
            img_size= self.img_size, 
            anchor_boxes=self.anchor_boxes, 
            k=self.k, 
            n_sample=self.n_sample, 
            backbone=self.backbone, 
            rpn_lambda=self.rpn_lambda,
            name='rpn'
            )
        self.get_candidate = get_candidate_layer(name='get_candidate')
        self.get_nms = NMS(iou_threshold=self.iou_threshold, name='get_nms')
        self.roipool = RoIpool(pool_size=self.pool_size, name='roipool')
        self.classifier = Classifier(name='classifier')

    def compile(self, rpn_optimizer, classifier_optimizer):
        self.rpn.compile(optimizer=rpn_optimizer)
        self.classifier.compile(optimizer=classifier_optimizer)
        super(Faster_RCNN, self).compile()

    def call(self, inputs):
        scores, rps, feature_map = self.rpn(inputs)
        rps = self.rpn.inverse_bbox_regression(rps)
        candidate_area, scores = self.get_candidate((scores, rps, self.n_test_pre_nms))
        nms = self.get_nms((candidate_area, scores, self.n_test_post_nms))
        rois = self.roipool((feature_map, nms))
        cls, bbox_reg, nms = self.classifier((rois, nms))
        predict = self.classifier.inverse_bbox_regression(bbox_reg, nms)
        return cls, predict