"""VGG19 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from keras import layers, models, utils
from utils.mish import Mish


WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
class VGG19:

    def __init__(self, input_shape=None, classes = 1000, activation = 'relu', include_top=True, weights='imagenet', pooling='avg'):
        self.init = layers.Input(input_shape)
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

       
        # Block 1
        x = layers.Conv2D(64, (3, 3),activation='relu',padding='same',
            name='block1_conv1')(self.init)

        x = layers.Conv2D(64, (3, 3),activation='relu',padding='same',
                        name='block1_conv2')(x)
                        
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3),activation='relu',padding='same',
                        name='block2_conv1')(x)

        x = layers.Conv2D(128, (3, 3),activation='relu',padding='same',
                        name='block2_conv2')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3),activation='relu',padding='same',
                        name='block3_conv1')(x)

        x = layers.Conv2D(256, (3, 3),activation='relu',padding='same',
                        name='block3_conv2')(x)

        x = layers.Conv2D(256, (3, 3),activation='relu',padding='same',
                        name='block3_conv3')(x)

        x = layers.Conv2D(256, (3, 3),activation='relu',padding='same',
                        name='block3_conv4')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),activation='relu',padding='same',
                        name='block4_conv1')(x)
        x = layers.Conv2D(512, (3, 3),activation='relu',padding='same',
                        name='block4_conv2')(x)
        x = layers.Conv2D(512, (3, 3),activation='relu',padding='same',
                        name='block4_conv3')(x)
        x = layers.Conv2D(512, (3, 3),activation='relu',padding='same',
                        name='block4_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3),activation='relu',padding='same',
                        name='block5_conv1')(x)
        x = layers.Conv2D(512, (3, 3),activation='relu',padding='same',
                        name='block5_conv2')(x)
        x = layers.Conv2D(512, (3, 3),activation='relu',padding='same',
                        name='block5_conv3')(x)
        x = layers.Conv2D(512, (3, 3),activation='relu',padding='same',
                        name='block5_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if self.include_top:
            # Classification block
            x = layers.Flatten(name='flatten')(x)
            x = layers.Dense(4096, activation='relu', name='fc1')(x)
            x = layers.Dense(4096, activation='relu', name='fc2')(x)
            x = layers.Dense(self.classes, activation='softmax', name='predictions')(x)
        else:
            if self.pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        inputs = self.init
        # Create model.
        model = models.Model(inputs, x, name='vgg19')

        # Load weights.
        if self.weights == 'imagenet':
            if self.include_top:
                weights_path = utils.get_file(
                    'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                    WEIGHTS_PATH,
                    cache_subdir='models',
                    file_hash='cbe5617147190e668d6c5d5026f83318')
            else:
                weights_path = utils.get_file(
                    'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='253f8cb515780f3b799900260a226db6')
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)

        return model