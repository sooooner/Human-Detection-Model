import tensorflow as tf
from utils.base_model import get_base
from utils.models.region_proposal_network import RPN
from utils.models.classifier import Classifier
from utils.models.layers import RoIPooling

class Faster_RCNN(tf.kersa.Model):
    def __init__(self, anchor_boxes, **kwargs):
        super(Faster_RCNN, self).__init__(*kwargs)
        self.anchor_boxes = anchor_boxes
        self.rpn = RPN()
        self.roipooling = RoIPooling()
        self.classifier = Classifier()
        self.rpn_train = False
        self.classifier_train = False

    def compile(self, rpn_optimizer, classifier_optimizer):
        super(Faster_RCNN, self).compile()
        self.rpn.optimizer = rpn_optimizer
        self.classifier.optimizer = classifier_optimizer
        
    def train_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        
        if self.rpn_train:
            self.classifier.trainable = False
            self.rpn.trainable = True
            result = self.rpn.train_step(x, (y_cls, y_reg))
            
        if self.classifier_train:
            self.classifier.trainable = True
            self.rpn.trainable = False
            scores, bboxes = self(x)
            result = self.classifier.train_step(x, (y_cls, y_reg))
        return result
    
    def test_step(self, data):
        x, y = data
        y_cls = y[0]
        y_reg = y[1]
        rpn_lambda = 5
        
        cls, bbox_reg, _ = self(x, training=False)
        cls_loss = self.Cls_Loss(y_cls, cls)
        reg_loss = self.Reg_Loss(y_reg, bbox_reg)
        losses = cls_loss + rpn_lambda * reg_loss

        self.test_loss_tracker.update_state(losses)
        return {'classifier_loss_val': self.test_loss_tracker.result()}
    
    # def call(self, inputs):
    #     scores, rps, feature_map = self.rpn(inputs)
    #     rois = get_rois(scores, rps, self.anchor_boxes)
    #     roi_projections_tensor, bboxes_reg_labels, labels = NMS(rois, scores, gts, feature_map)
    #     scores, bboxes = self.classifier(roi_projections_tensor)
    #     return scores, bboxes