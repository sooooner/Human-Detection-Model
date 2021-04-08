import tensorflow as tf
import numpy as np

class Classifier(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(2048, 7, 7, name='cls_conv')
        self.flatten = tf.keras.layers.Flatten(name='cls_flatten')
        self.dense = tf.keras.layers.Dense(2048, name='cls_dense')
        self.cls = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_out')
        self.bbox_reg = tf.keras.layers.Dense(4, name='bbox_out')

        self.deconv = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=80, kernel_size=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(80, 14, 14)
        self.flatten2 = tf.keras.layers.Flatten()
        self.mask_out = tf.keras.layers.Dense(48, activation='sigmoid')

    def compile(self, optimizer):
        super(Classifier, self).compile()
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.test_loss_tracker = tf.keras.metrics.Mean(name='test_loss')
    
    def Cls_Loss(self, y_true, y_pred):
        return tf.losses.BinaryCrossentropy()(y_true, y_pred)

    def Reg_Loss(self, y_true, y_pred, indices):
        return tf.losses.Huber()(y_true[indices], y_pred[indices])

    def Mask_Loss(self, y_true, y_pred, indices):
        y_true = tf.reshape(tf.tile(y_true, [1, indices.shape[1]]), (-1, indices.shape[1], 48))
        return tf.keras.losses.MSE(y_true[indices], y_pred[indices])
    
    def train_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        y_mask = y[2]
        indices = tf.not_equal(y_cls, 0)
        
        with tf.GradientTape() as tape:
            cls, bbox_reg, mask, _ = self(x)
            cls_loss = self.Cls_Loss(y_cls, cls)
            reg_loss = self.Reg_Loss(y_reg, bbox_reg, indices)
            mask_loss = self.Mask_Loss(y_mask, mask, indices)
            losses = cls_loss + reg_loss + mask_loss
            
        trainable_vars = self.trainable_variables
        grad = tape.gradient(losses, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))
        self.loss_tracker.update_state(losses)
        return {'classifier_loss': self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        y_mask = y[2]
        indices = tf.not_equal(y_cls, 0)

        cls, bbox_reg, mask, _ = self(x, training=False)
        cls_loss = self.Cls_Loss(y_cls, cls)
        reg_loss = self.Reg_Loss(y_reg, bbox_reg, indices)
        mask_loss = self.Mask_Loss(y_mask, mask, indices)
        losses = cls_loss + reg_loss + mask_loss

        self.test_loss_tracker.update_state(losses)
        return {'classifier_loss_val': self.test_loss_tracker.result()}

    def bbox_regression(self, bbox, nmses):
        tx = (bbox[:, :, 0] - nmses[:, :, 0]) / nmses[:, :, 2]
        ty = (bbox[:, :, 1] - nmses[:, :, 1]) / nmses[:, :, 3]
        tw = tf.math.log(tf.maximum(bbox[:, :, 2], np.finfo(np.float64).eps) / nmses[:, :, 2])
        th = tf.math.log(tf.maximum(bbox[:, :, 3], np.finfo(np.float64).eps) / nmses[:, :, 3])
        return tf.stack([tx, ty, tw, th], -1)

    @staticmethod
    def inverse_bbox_regression(bbox, nmses):
        gx = nmses[:, :, 2] * bbox[:, :, 0] + nmses[:, :, 0]
        gy = nmses[:, :, 3] * bbox[:, :, 1] + nmses[:, :, 1]
        gw = nmses[:, :, 2] * tf.exp(bbox[:, :, 2])
        gh = nmses[:, :, 3] * tf.exp(bbox[:, :, 3])
        return tf.stack([gx, gy, gw, gh], -1)

    def call(self, inputs):
        rois, nms = inputs

        x = tf.keras.layers.TimeDistributed(self.deconv)(rois)
        x = tf.keras.layers.TimeDistributed(self.bn)(x)
        x = tf.keras.layers.TimeDistributed(self.relu)(x) 

        x = tf.keras.layers.TimeDistributed(self.conv1)(x)
        x = tf.keras.layers.TimeDistributed(self.bn1)(x)
        x = tf.keras.layers.TimeDistributed(self.relu1)(x) 

        x = tf.keras.layers.TimeDistributed(self.conv2)(x)
        x = tf.keras.layers.TimeDistributed(self.flatten2)(x) 
        mask = tf.keras.layers.TimeDistributed(self.mask_out)(x)

        x = tf.keras.layers.TimeDistributed(self.conv)(rois)
        x = tf.keras.layers.TimeDistributed(self.flatten)(x)
        feature_vector = tf.keras.layers.TimeDistributed(self.dense)(x)
        clss = tf.keras.layers.TimeDistributed(self.cls)(feature_vector)
        bbox = tf.keras.layers.TimeDistributed(self.bbox_reg)(feature_vector)
        bbox_reg = self.bbox_regression(bbox, nms)

        return clss, bbox_reg, mask, nms