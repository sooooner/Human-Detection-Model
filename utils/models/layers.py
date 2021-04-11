import tensorflow as tf

class get_candidate_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(get_candidate_layer, self).__init__(**kwargs)

    def anchors_clip(self, boxes, size=(432, 768)):    
        x1 = boxes[:, :, 0] - boxes[:, :, 2]/2
        x2 = boxes[:, :, 0] + boxes[:, :, 2]/2
        y1 = boxes[:, :, 1] - boxes[:, :, 3]/2
        y2 = boxes[:, :, 1] + boxes[:, :, 3]/2
        
        x1 = tf.clip_by_value(x1, 0, size[1])
        x2 = tf.clip_by_value(x2, 0, size[1])
        y1 = tf.clip_by_value(y1, 0, size[0])
        y2 = tf.clip_by_value(y2, 0, size[0])

        w = x2 - x1
        h = y2 - y1
        x = x1 + w/2
        y = y1 + h/2
        return tf.stack([x, y, w, h], axis=-1)

    def call(self, x):
        scores, rps, n_train_pre_nms = x
        rois = self.anchors_clip(rps)

        oobw = tf.expand_dims(tf.cast(tf.math.greater(rois[:, :, 2], 16), tf.float32), -1)
        oobh = tf.expand_dims(tf.cast(tf.math.greater(rois[:, :, 3], 16), tf.float32), -1)
        scores = tf.math.multiply(scores, oobw)
        scores = tf.math.multiply(scores, oobh)

        orders = tf.argsort(scores, direction='DESCENDING', axis=1)[:, :n_train_pre_nms]
        rois = tf.gather_nd(rois, orders, batch_dims=1)
        scores = tf.gather_nd(scores, orders, batch_dims=1)
        return rois, scores

class NMS(tf.keras.layers.Layer):
    def __init__(self, iou_threshold=0.7, **kwargs):
        self.iou_threshold = iou_threshold
        super(NMS, self).__init__(**kwargs)

    def call(self, inputs):
        rois, scores, max_output_size = inputs
        selected_indices_padded = tf.image.non_max_suppression_padded(
            rois, 
            tf.squeeze(scores), 
            max_output_size=max_output_size,
            iou_threshold=0.7,
            pad_to_max_output_size=True
        )[0]
        nms = tf.gather(rois, selected_indices_padded, batch_dims=1)
        return nms

class RoIpool(tf.keras.layers.Layer):
    def __init__(self, pool_size=7, **kwargs):
        self.pool_size = pool_size
        super(RoIpool, self).__init__(**kwargs)

    def cal_rois_ratio(self, nmses, size=[432, 768]):
        x1 = (nmses[:, :, 0] - nmses[:, :, 2]/2)/size[1]
        x2 = (nmses[:, :, 0] + nmses[:, :, 2]/2)/size[1]
        y1 = (nmses[:, :, 1] - nmses[:, :, 3]/2)/size[0]
        y2 = (nmses[:, :, 1] + nmses[:, :, 3]/2)/size[0]
        return tf.stack([y1, x1, y2, x2], axis=-1)

    def call(self, inputs):
        # feature_map, nmses, batch_size = inputs
        feature_map, nmses = inputs
        n_channel = feature_map.shape[-1]
        batch_size = nmses.shape[0]
        num_rois = nmses.shape[1]
        if batch_size is None:
            batch_size = 16
        nmses = self.cal_rois_ratio(nmses)
        rois = tf.image.crop_and_resize(
            feature_map, 
            tf.reshape(nmses, (-1, 4)), 
            box_indices=[i for i in range(batch_size) for _ in range(num_rois)], 
            crop_size=[self.pool_size, self.pool_size]
        )
        return tf.reshape(rois, shape=(batch_size, num_rois, self.pool_size, self.pool_size, n_channel))