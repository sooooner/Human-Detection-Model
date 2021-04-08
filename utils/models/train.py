import tensorflow as tf
from IPython.display import display
from utils.label_generator import classifier_label_generator
import time

def frcnn_train_step(model, train_dataset, train_stage, epochs=1, valid_dataset=None, change_lr=False, rpn_lr=None, cls_lr=None):
    if change_lr:
        if rpn_lr:
            tf.keras.backend.set_value(model.rpn.optimizer.learning_rate, rpn_lr)
        if cls_lr:
            tf.keras.backend.set_value(model.classifier.optimizer.learning_rate, cls_lr)

    if train_stage == 1:
        print('Train RPNs \n')
        model.rpn.trainable = True
        model.classifier.trainable = False
    elif train_stage == 2:
        print('Train Fast R-CNN using the proposals from RPNs \n')
        model.rpn.trainable = False
        model.rpn.base_model.trainable = True
        model.classifier.trainable = True
    elif train_stage == 3:
        print('Fix the shared convolutional layers and fine-tune unique layers to RPN \n')
        model.rpn.trainable = True
        model.rpn.base_model.trainable = False
        model.classifier.trainable = False
    elif train_stage == 4:
        print('Fine-tune unique layers to Fast R-CNN \n')
        model.rpn.trainable = False
        model.classifier.trainable = True

    max_step = 'Unknown'
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"epoch {epoch+1}/{epochs}")
        display_loss = display("Training loss (for one batch) at step 0 : 0", display_id=True)
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            start = time.time()
            y_cls_rpn = y_batch_train[0]
            y_reg_rpn = y_batch_train[1]
            gts = y_batch_train[2]
            mask = y_batch_train[3]
            
            if train_stage == 1 or train_stage == 3:
                result = model.rpn.train_step((x_batch_train, (y_cls_rpn, y_reg_rpn)))
                losses = round(result['rpn_loss'].numpy(), 4)
            else:
                scores, rps, feature_map = model.rpn(x_batch_train)
                rps = model.rpn.inverse_bbox_regression(rps)
                candidate_area, scores = model.get_candidate((scores, rps, model.n_train_pre_nms))
                nms = model.get_nms((candidate_area, scores, model.n_train_post_nms))
                box_labels, cls_labels, nms = classifier_label_generator(nms, gts)
                rois = model.roipool((feature_map, nms))
                result = model.classifier.train_step(((rois, nms), (cls_labels, box_labels, mask)))
                losses = round(result['classifier_loss'].numpy(), 4)

            display_loss.update(f"Training loss at step {step}/{max_step} : {losses} - {round(time.time() - start, 4)}sec/step")
            
        max_step = step
        display_loss.update(f"Training loss at step {step}/{max_step} : {losses} - {round(time.time()-start, 4)}sec/step - {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time()-epoch_start))}/epoch")

        if valid_dataset is not None:
            display_loss_valid = display("validation loss : 0", display_id=True)
            for x_batch_test, y_batch_test in valid_dataset:
                start = time.time()
                y_cls_rpn = y_batch_test[0]
                y_reg_rpn = y_batch_test[1]
                gts = y_batch_test[2]
                mask = y_batch_test[3]

                if train_stage == 1 or train_stage == 3:
                    result = model.rpn.train_step((x_batch_test, (y_cls_rpn, y_reg_rpn)))
                    losses = result['rpn_loss'].numpy()
                else:
                    scores, rps, feature_map = model.rpn(x_batch_test)
                    rps = model.rpn.inverse_bbox_regression(rps)
                    candidate_area, scores = model.get_candidate((scores, rps, model.n_test_pre_nms))
                    nms = model.get_nms((candidate_area, scores, model.n_test_post_nms))
                    box_labels, cls_labels, nms = classifier_label_generator(nms, gts, valid=True)
                    rois = model.roipool((feature_map, nms))
                    result = model.classifier.train_step(((rois, nms), (cls_labels, box_labels, mask)))
                    losses = result['classifier_loss'].numpy()
                
            display_loss_valid.update(f"validation loss : {losses}")

    return model