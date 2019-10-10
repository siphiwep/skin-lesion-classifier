from __future__ import print_function
import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from custom_layers.pool_helper import PoolHelper
from custom_layers.lrn_layer import LRN
from custom_layers.spatial_pyramid_pooling import SpatialPyramidPooling
# K.set_image_data_format('channels_first')
if keras.backend.backend() == 'tensorflow':
    from keras import backend as K
    import tensorflow as tf
    from keras.utils.conv_utils import convert_kernel

class GoogleNet:

    def __init__(self, input_shape, classes,  weights_path=None):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
        self.weights_path = weights_path
        self.input = Input(shape=(input_shape))
        self.classes = classes

    def inception(self, x, filters):
        # 1x1
        path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)

        # 1x1->3x3
        path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
        path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)
        
        # 1x1->5x5
        path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
        path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)

        # 3x3->1x1
        path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
        path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)

        return Concatenate(axis=-1)([path1,path2,path3,path4])


    def auxiliary(self, x, name=None):
        layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
        layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
        layer = Flatten()(layer)
        layer = Dense(units=256, activation='relu')(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(units=self.classes, activation='softmax', name=name)(layer)
        return layer


    def model(self):
        # stage-1
        layer = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(self.input)
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = BatchNormalization()(layer)

        # stage-2
        layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
        layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)

        # stage-3
        layer = self.inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
        layer = self.inception(layer, [128, (128,192), (32,96), 64]) #3b
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        
        # stage-4
        layer = self.inception(layer, [192,  (96,208),  (16,48),  64]) #4a
        aux1  = self.auxiliary(layer, name='aux1')
        layer = self.inception(layer, [160, (112,224),  (24,64),  64]) #4b
        layer = self.inception(layer, [128, (128,256),  (24,64),  64]) #4c
        layer = self.inception(layer, [112, (144,288),  (32,64),  64]) #4d
        aux2  = self.auxiliary(layer, name='aux2')
        layer = self.inception(layer, [256, (160,320), (32,128), 128]) #4e
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        
        # stage-5
        layer = self.inception(layer, [256, (160,320), (32,128), 128]) #5a
        layer = self.inception(layer, [384, (192,384), (48,128), 128]) #5b
        layer = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)
        
        # stage-6
        layer = Flatten()(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(units=256, activation='relu')(layer)
        main = Dense(units=self.classes, activation='softmax', name='main')(layer)
        
        model = Model(inputs=self.input, outputs=[main, aux1, aux2], name="googlenet")
        
        return model