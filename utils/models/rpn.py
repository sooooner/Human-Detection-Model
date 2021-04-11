import tensorflow as tf
import numpy as np
from utils.models.base_model import get_base

class RPN(tf.keras.models.Model):
    def __init__(self, img_size, anchor_boxes, k=5*5, n_sample=32, backbone='resnet50', rpn_lambda=10, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self.img_size = img_size
        self.anchor_boxes = anchor_boxes
        self.num_of_anchor = len(self.anchor_boxes)
        self.n_sample = n_sample
        self.k = k
        self.backbone = backbone
        self.rpn_lambda = rpn_lambda

        self.base_model = get_base(self.img_size, model=self.backbone)
        self.window = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='window')
        self.window_bn = tf.keras.layers.BatchNormalization()
        self.window_relu = tf.keras.layers.ReLU()

        self.bbox_reg = tf.keras.layers.Conv2D(filters=self.k*4, kernel_size=1, name='bbox_reg')
        self.bbox_reg_reshape = tf.keras.layers.Reshape((-1, 4), name='reg_out')
        self.cls = tf.keras.layers.Conv2D(filters=self.k, kernel_size=1, activation='sigmoid', name='cls')
        self.cls_reshape = tf.keras.layers.Reshape((-1, 1), name='cls_out')

    def compile(self, optimizer, **kwargs):
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.test_loss_tracker = tf.keras.metrics.Mean(name='test_loss')
        self.optimizer = optimizer
        super(RPN, self).compile(**kwargs)
    
    def Cls_Loss(self, y_true, y_pred):
        indices = tf.where(tf.not_equal(y_true, tf.constant(-1.0, dtype=tf.float32)))
        target = tf.gather_nd(y_true, indices)
        output = tf.gather_nd(y_pred, indices)
        return tf.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.SUM)(target, output)/self.n_sample

    def Reg_Loss(self, y_true, y_pred):
        indices = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)
        return tf.losses.Huber(reduction=tf.losses.Reduction.SUM)(y_true[indices], y_pred[indices])/self.num_of_anchor

    def train_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        
        with tf.GradientTape() as tape:
            cls, bbox_reg, _ = self(x, training=True)
            cls_loss = self.Cls_Loss(y_cls, cls)
            reg_loss = self.Reg_Loss(y_reg, bbox_reg)
            losses = cls_loss + self.rpn_lambda * reg_loss
            
        trainable_vars = self.trainable_variables
        grad = tape.gradient(losses, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))
        self.loss_tracker.update_state(losses)
        return {'rpn_loss': self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        
        cls, bbox_reg, _ = self(x, training=False)
        cls_loss = self.Cls_Loss(y_cls, cls)
        reg_loss = self.Reg_Loss(y_reg, bbox_reg)
        losses = cls_loss + self.rpn_lambda * reg_loss

        self.test_loss_tracker.update_state(losses)
        return {'rpn_loss_val': self.test_loss_tracker.result()}

    @tf.function
    def bbox_regression(self, boxes):
        tx = (boxes[:, :, 0] - self.anchor_boxes[:, 0]) / self.anchor_boxes[:, 2]
        ty = (boxes[:, :, 1] - self.anchor_boxes[:, 1]) / self.anchor_boxes[:, 3]
        tw = tf.math.log(tf.maximum(boxes[:, :, 2], np.finfo(np.float64).eps) / self.anchor_boxes[:, 2])
        th = tf.math.log(tf.maximum(boxes[:, :, 3], np.finfo(np.float64).eps) / self.anchor_boxes[:, 3])
        return tf.stack([tx, ty, tw, th], -1)

    @tf.function
    def inverse_bbox_regression(self, boxes):
        gx = self.anchor_boxes[:, 2] * boxes[:, :, 0] + self.anchor_boxes[:, 0]
        gy = self.anchor_boxes[:, 3] * boxes[:, :, 1] + self.anchor_boxes[:, 1]
        gw = self.anchor_boxes[:, 2] * tf.exp(boxes[:, :, 2])
        gh = self.anchor_boxes[:, 3] * tf.exp(boxes[:, :, 3])
        return tf.stack([gx, gy, gw, gh], axis=-1)
        
    def call(self, inputs):
        feature_extractor = self.base_model(inputs)
        intermediate = self.window(feature_extractor)
        intermediate = self.window_bn(intermediate)
        intermediate = self.window_relu(intermediate)

        cls = self.cls(intermediate)
        cls = self.cls_reshape(cls)
        bbox_reg = self.bbox_reg(intermediate)
        bbox_reg = self.bbox_reg_reshape(bbox_reg)
        bbox_reg = self.bbox_regression(bbox_reg)
        return cls, bbox_reg, feature_extractor