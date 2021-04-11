import tensorflow as tf
import numpy as np

class Classifier(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1024, kernel_size=(7, 7), name='conv')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten(name='flatten')

        self.dense1 = tf.keras.layers.Dense(1024, name='dense1')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.dense2 = tf.keras.layers.Dense(1024, name='dense2')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.cls_dense = tf.keras.layers.Dense(256, name='cls_dense')
        self.cls_bn = tf.keras.layers.BatchNormalization()
        self.cls_relu = tf.keras.layers.ReLU()

        self.bbox_dense = tf.keras.layers.Dense(256, name='bbox_dense')
        self.bbox_bn = tf.keras.layers.BatchNormalization()
        self.bbox_relu = tf.keras.layers.ReLU()

        self.cls = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_out')
        self.bbox_reg = tf.keras.layers.Dense(4, name='bbox_out')

        # # mask brunch
        # self.mask1_deconv = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=2)
        # self.mask1_bn = tf.keras.layers.BatchNormalization()
        # self.mask1_relu = tf.keras.layers.ReLU()

        # self.mask3_deconv = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2)
        # self.mask3_bn = tf.keras.layers.BatchNormalization()
        # self.mask3_relu = tf.keras.layers.ReLU()

        # self.mask4_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(28, 28))
        # self.mask4_bn = tf.keras.layers.BatchNormalization()
        # self.mask4_relu = tf.keras.layers.ReLU()
        # self.mask4_flatten = tf.keras.layers.Flatten()

        # self.mask_out = tf.keras.layers.Dense(48, activation='sigmoid')

    def compile(self, optimizer):
        super(Classifier, self).compile()
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.test_loss_tracker = tf.keras.metrics.Mean(name='test_loss')
    
    def Cls_Loss(self, y_true, y_pred):
        return tf.losses.BinaryCrossentropy()(y_true, y_pred)

    def Reg_Loss(self, y_true, y_pred, indices):
        return tf.losses.Huber()(y_true[indices], y_pred[indices])

    # def Mask_Loss(self, y_true, y_pred, indices):
    #     y_true = tf.reshape(tf.tile(y_true, [1, indices.shape[1]]), (-1, indices.shape[1], 48))
    #     return tf.keras.losses.MSE(y_true[indices], y_pred[indices])

    def train_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        # y_mask = y[2]
        indices = tf.not_equal(y_cls, 0)
        
        with tf.GradientTape() as tape:
            # cls, bbox_reg, mask, _ = self(x, training=True)
            cls, bbox_reg, _ = self(x, training=True)
            cls_loss = self.Cls_Loss(y_cls, cls)
            reg_loss = self.Reg_Loss(y_reg, bbox_reg, indices)
            # mask_loss = self.Mask_Loss(y_mask, mask, indices)
            # losses = cls_loss + reg_loss + mask_loss
            losses = cls_loss + reg_loss 
            
        trainable_vars = self.trainable_variables
        grad = tape.gradient(losses, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))
        self.loss_tracker.update_state(losses)
        return {'classifier_loss': self.loss_tracker.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        # y_mask = y[2]
        indices = tf.not_equal(y_cls, 0)

        # cls, bbox_reg, mask, _ = self(x, training=False)
        cls, bbox_reg, _ = self(x, training=False)
        cls_loss = self.Cls_Loss(y_cls, cls)
        reg_loss = self.Reg_Loss(y_reg, bbox_reg, indices)
        # mask_loss = self.Mask_Loss(y_mask, mask, indices)
        # losses = cls_loss + reg_loss + mask_loss
        losses = cls_loss + reg_loss 

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

    @tf.function
    def call(self, inputs, training=False):
        rois, nms = inputs

        cls_x = tf.keras.layers.TimeDistributed(self.conv)(rois)
        cls_x = tf.keras.layers.TimeDistributed(self.bn)(cls_x)
        cls_x = tf.keras.layers.TimeDistributed(self.relu)(cls_x)
        cls_x = tf.keras.layers.TimeDistributed(self.flatten)(cls_x)

        cls_x = tf.keras.layers.TimeDistributed(self.dense1)(cls_x)
        cls_x = tf.keras.layers.TimeDistributed(self.bn1)(cls_x)
        cls_x = tf.keras.layers.TimeDistributed(self.relu1)(cls_x)
        
        cls_x = tf.keras.layers.TimeDistributed(self.dense2)(cls_x)
        cls_x = tf.keras.layers.TimeDistributed(self.bn2)(cls_x)
        feature_vector = tf.keras.layers.TimeDistributed(self.relu2)(cls_x)

        cls_x = tf.keras.layers.TimeDistributed(self.cls_dense)(feature_vector)
        cls_x = tf.keras.layers.TimeDistributed(self.cls_bn)(cls_x)
        cls_x = tf.keras.layers.TimeDistributed(self.cls_relu)(cls_x)

        bbox_x = tf.keras.layers.TimeDistributed(self.bbox_dense)(feature_vector)
        bbox_x = tf.keras.layers.TimeDistributed(self.bbox_bn)(bbox_x)
        bbox_x = tf.keras.layers.TimeDistributed(self.bbox_relu)(bbox_x)

        clss = tf.keras.layers.TimeDistributed(self.cls)(cls_x)
        bbox = tf.keras.layers.TimeDistributed(self.bbox_reg)(bbox_x)
        bbox_reg = self.bbox_regression(bbox, nms)

        # # mask brunch
        # mask_x = tf.keras.layers.TimeDistributed(self.mask1_deconv)(rois)
        # mask_x = tf.keras.layers.TimeDistributed(self.mask1_bn)(mask_x)
        # mask_x = tf.keras.layers.TimeDistributed(self.mask1_relu)(mask_x) 

        # mask_x = tf.keras.layers.TimeDistributed(self.mask3_deconv)(mask_x)
        # mask_x = tf.keras.layers.TimeDistributed(self.mask3_bn)(mask_x) 
        # mask_x = tf.keras.layers.TimeDistributed(self.mask3_relu)(mask_x)

        # mask_x = tf.keras.layers.TimeDistributed(self.mask4_conv)(mask_x)
        # mask_x = tf.keras.layers.TimeDistributed(self.mask4_bn)(mask_x) 
        # mask_x = tf.keras.layers.TimeDistributed(self.mask4_relu)(mask_x)
        # mask_x = tf.keras.layers.TimeDistributed(self.mask4_flatten)(mask_x) 
        # mask = tf.keras.layers.TimeDistributed(self.mask_out)(mask_x)
        # return clss, bbox_reg, mask, nms

        return clss, bbox_reg, nms