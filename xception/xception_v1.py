"""Xception V1 model for Keras.
On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.
Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).
# Reference
- [Xception: Deep Learning with Depthwise Separable Convolutions](
    https://arxiv.org/abs/1610.02357) (CVPR 2017)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from keras import layers, models, utils
from keras.applications import imagenet_utils
from utils.mish import Mish


TF_WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels.h5')
TF_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


class Xception:

    def __init__(self, input_shape=None, input_tensor=None, classes = 1000, activation = 'relu', include_top=True, weights='imagenet', pooling='avg'):
        self.init = layers.Input(shape=input_shape)
        self.classes = classes
        self.activation = activation
        self.include_top = include_top
        self.weights = weights
        self.pooling = pooling

    def model(self):
        if not (self.weights in {'imagenet', None} or os.path.exists(self.weights)):
            raise ValueError('The `weights` argument should be either '
                            '`None` (random initialization), `imagenet` '
                            '(pre-training on ImageNet), '
                            'or the path to the weights file to be loaded.')

        if self.weights == 'imagenet' and self.include_top and self.classes != 1000:
            raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                            ' as true, `classes` should be 1000')

        channel_axis = 3 

        x = layers.Conv2D(32, (3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        name='block1_conv1')(self.init)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = layers.Activation(self.activation, name='block1_conv1_act')(x)
        x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = layers.Activation(self.activation, name='block1_conv2_act')(x)

        residual = layers.Conv2D(128, (1, 1),
                                strides=(2, 2),
                                padding='same',
                                use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.SeparableConv2D(128, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block2_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = layers.Activation(self.activation, name='block2_sepconv2_act')(x)
        x = layers.SeparableConv2D(128, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block2_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                                padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation(self.activation, name='block3_sepconv1_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block3_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = layers.Activation(self.activation, name='block3_sepconv2_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block3_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(728, (1, 1),
                                strides=(2, 2),
                                padding='same',
                                use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation(self.activation, name='block4_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block4_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = layers.Activation(self.activation, name='block4_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block4_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.Activation(self.activation, name=prefix + '_sepconv1_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv1_bn')(x)
            x = layers.Activation(self.activation, name=prefix + '_sepconv2_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv2_bn')(x)
            x = layers.Activation(self.activation, name=prefix + '_sepconv3_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv3')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv3_bn')(x)

            x = layers.add([x, residual])

        residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                                padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation(self.activation, name='block13_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block13_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = layers.Activation(self.activation, name='block13_sepconv2_act')(x)
        x = layers.SeparableConv2D(1024, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block13_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block13_pool')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(1536, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block14_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = layers.Activation(self.activation, name='block14_sepconv1_act')(x)

        x = layers.SeparableConv2D(2048, (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block14_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = layers.Activation(self.activation, name='block14_sepconv2_act')(x)

        if self.include_top:
            #Classification block
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(self.classes, activation='softmax', name='predictions')(x)
        else:
            if self.pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`
        inputs = self.init
            # Create model.
        model = models.Model(inputs, x, name='xception')

        # Load weights.
        if self.weights == 'imagenet':
            if self.include_top:
                weights_path = utils.get_file(
                    'xception_weights_tf_dim_ordering_tf_kernels.h5',
                    TF_WEIGHTS_PATH,
                    cache_subdir='models',
                    file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
            else:
                weights_path = utils.get_file(
                    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    TF_WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b0042744bf5b25fce3cb969f33bebb97')
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)

        return model


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)