import tensorflow as tf

def get_base(img_size, model='resnet50'):
    if model=='vgg':
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=img_size)
        feature_extractor = base_model.get_layer("block5_conv3")
    elif model == 'resnet101':
        base_model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet', input_shape=img_size)
        feature_extractor = base_model.get_layer("conv4_block23_out")
    elif model == 'resnet50':
        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=img_size)
        feature_extractor = base_model.get_layer("conv4_block6_out")  
    else:
        raise Exception('you can choose only one of vgg, resnet50, or resnet101.')
        
    base_model = tf.keras.models.Model(inputs=base_model.input, outputs=feature_extractor.output)
    return base_model