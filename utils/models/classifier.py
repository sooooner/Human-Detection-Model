import tensorflow as tf
import numpy as np

def resnet_C5():
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg')
    roi_inputs = tf.keras.Input(shape=(7, 7, 1024))

    first_layer = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1))
    first_layer(roi_inputs)
    first_layer.set_weights(base_model.layers[143].get_weights())

    x = first_layer(roi_inputs)
    for i in range(144, 148+1):
        x = base_model.layers[i](x)
    x = base_model.layers[150](x)
    x = base_model.layers[152](x)

    short_cut_layer = tf.keras.layers.Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1))
    short_cut_layer(roi_inputs)
    short_cut_layer.set_weights(base_model.layers[149].get_weights())

    short_cut = short_cut_layer(roi_inputs)
    short_cut = base_model.layers[151](short_cut)
    x = base_model.layers[153]([x, short_cut])
    short_cut = base_model.layers[154](x)

    for i in range(155, 162+1):
        x = base_model.layers[i](x)
    x = base_model.layers[163]([x, short_cut])
    short_cut = base_model.layers[164](x)

    for i in range(165, 172+1):
        x = base_model.layers[i](x)
    x = base_model.layers[173]([x, short_cut])
    x = base_model.layers[174](x)
    output = base_model.layers[175](x)
    return tf.keras.models.Model(inputs=roi_inputs, outputs=output)

class Classifier(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.cls_lambda = 1
        self.res5 = resnet_C5()

        # self.conv = tf.keras.layers.Conv2D(2048, kernel_size=(7, 7), name='conv')
        # self.bn = tf.keras.layers.BatchNormalization(name='conv_bn')
        # self.relu = tf.keras.layers.ReLU(name='conv_relu')
        # self.flatten = tf.keras.layers.Flatten(name='flatten')

        # self.dense1 = tf.keras.layers.Dense(1024, name='dense1')
        # self.bn1 = tf.keras.layers.BatchNormalization(name='dense1_bn')
        # self.relu1 = tf.keras.layers.ReLU(name='dense1_relu')

        # self.dense2 = tf.keras.layers.Dense(1024, name='dense2')
        # self.bn2 = tf.keras.layers.BatchNormalization(name='dense2_bn')
        # self.relu2 = tf.keras.layers.ReLU(name='dense2_relu')

        self.cls_dense = tf.keras.layers.Dense(256, name='cls_dense')
        self.cls_bn = tf.keras.layers.BatchNormalization(name='cls_bn')
        self.cls_relu = tf.keras.layers.ReLU(name='cls_relu')
        self.cls = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_out')

        self.bbox_dense = tf.keras.layers.Dense(256, name='bbox_dense')
        self.bbox_bn = tf.keras.layers.BatchNormalization(name='bbox_bn')
        self.bbox_relu = tf.keras.layers.ReLU(name='bbox_relu')
        self.bbox_reg = tf.keras.layers.Dense(4, name='bbox_out')

    def compile(self, optimizer):
        super(Classifier, self).compile()
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.test_loss_tracker = tf.keras.metrics.Mean(name='test_loss')
    

    def Cls_Loss(self, y_true, y_pred):
        return tf.losses.BinaryCrossentropy()(y_true, y_pred)

    def Reg_Loss(self, y_true, y_pred, indices):
        return tf.losses.Huber()(y_true[indices], y_pred[indices])

    @tf.function
    def train_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        indices = tf.not_equal(y_cls, 0)
        
        with tf.GradientTape() as tape:
            cls, bbox_reg, _ = self(x, training=True)
            cls_loss = self.Cls_Loss(y_cls, cls)
            reg_loss = self.Reg_Loss(y_reg, bbox_reg, indices)
            losses = cls_loss + self.cls_lambda * reg_loss 
            
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
        indices = tf.not_equal(y_cls, 0)

        cls, bbox_reg, _ = self(x, training=False)
        cls_loss = self.Cls_Loss(y_cls, cls)
        reg_loss = self.Reg_Loss(y_reg, bbox_reg, indices)
        losses = cls_loss + self.cls_lambda * reg_loss 

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
        feature_vector = tf.keras.layers.TimeDistributed(self.res5)(rois)

        # cls_x = tf.keras.layers.TimeDistributed(self.conv)(rois)
        # cls_x = tf.keras.layers.TimeDistributed(self.bn)(cls_x)
        # cls_x = tf.keras.layers.TimeDistributed(self.relu)(cls_x)
        # cls_x = tf.keras.layers.TimeDistributed(self.flatten)(cls_x)

        # cls_x = tf.keras.layers.TimeDistributed(self.dense1)(cls_x)
        # cls_x = tf.keras.layers.TimeDistributed(self.bn1)(cls_x)
        # cls_x = tf.keras.layers.TimeDistributed(self.relu1)(cls_x)
        
        # cls_x = tf.keras.layers.TimeDistributed(self.dense2)(cls_x)
        # cls_x = tf.keras.layers.TimeDistributed(self.bn2)(cls_x)
        # feature_vector = tf.keras.layers.TimeDistributed(self.relu2)(cls_x)

        cls_x = tf.keras.layers.TimeDistributed(self.cls_dense)(feature_vector)
        cls_x = tf.keras.layers.TimeDistributed(self.cls_bn)(cls_x)
        cls_x = tf.keras.layers.TimeDistributed(self.cls_relu)(cls_x)
        clss = tf.keras.layers.TimeDistributed(self.cls)(cls_x)
        
        bbox_x = tf.keras.layers.TimeDistributed(self.bbox_dense)(feature_vector)
        bbox_x = tf.keras.layers.TimeDistributed(self.bbox_bn)(bbox_x)
        bbox_x = tf.keras.layers.TimeDistributed(self.bbox_relu)(bbox_x)
        bbox = tf.keras.layers.TimeDistributed(self.bbox_reg)(bbox_x)
        bbox_reg = self.bbox_regression(bbox, nms)

        return clss, bbox_reg, nms