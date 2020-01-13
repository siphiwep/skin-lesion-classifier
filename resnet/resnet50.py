"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from keras import layers, models, utils
from utils.mish import Mish
# from . import get_submodules_from_kwargs
# from . import imagenet_utils
# from .imagenet_utils import decode_predictions
# from .imagenet_utils import _obtain_input_shape

# preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

class  ResNet50:

    def __init__(self, input_shape=None, classes = 1000, activation = 'relu', include_top=True, weights='imagenet', pooling='avg'):
        self.init = layers.Input(input_shape)
        self.classes = classes
        self.activation = activation
        self.include_top = include_top
        self.weights = weights
        self.pooling = pooling
    
    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        bn_axis = 3
     
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(filters2, kernel_size,
                        padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation(self.activation)(x)
        return x

    def conv_block(self,input_tensor, kernel_size,filters,stage,block,strides=(2, 2)):
      
        filters1, filters2, filters3 = filters
  
        bn_axis = 3
        
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1), strides=strides,
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                kernel_initializer='he_normal',
                                name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation(self.activation)(x)
        return x


    def model(self):
    
        # global backend, layers, models, keras_utils
        # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

        if not (self.weights in {'imagenet', None} or os.path.exists(self.weights)):
            raise ValueError('The `weights` argument should be either '
                            '`None` (random initialization), `imagenet` '
                            '(pre-training on ImageNet), '
                            'or the path to the weights file to be loaded.')

        if self.weights == 'imagenet' and self.include_top and self.classes != 1000:
            raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                            ' as true, `classes` should be 1000')
        
        bn_axis = 3

        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(self.init)
        x = layers.Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer='he_normal',
                        name='conv1')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = layers.Activation(self.activation)(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        if self.include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(self.classes, activation='softmax', name='fc1000')(x)
        else:
            if self.pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)
            else:
                warnings.warn('The output shape of `ResNet50(include_top=False)` '
                            'has been changed since Keras 2.2.0.')

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
    
        inputs = self.init
        # Create model.
        model = models.Model(inputs, x, name='resnet50')

        # Load weights.
        if self.weights == 'imagenet':
            if self.include_top:
                weights_path = utils.get_file(
                    'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                    WEIGHTS_PATH,
                    cache_subdir='models',
                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = utils.get_file(
                    'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)

        return model