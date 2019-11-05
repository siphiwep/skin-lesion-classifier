"""
    Author: Pike Msonda
    Description: AlexNet implementation using Keras api
"""

from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(1000)
np.random.seed(1000)
random.seed(1000)
REGULARIZER= 0.001
class AlexNet:

    def __init__(self, input_shape, classes, weights_path=''):
        self.init = Input(input_shape)
        self.classes = classes
        self.weights_path = weights_path

    def conv_layer(self, x, filters,kernel_size, padding= "same", 
            kernel_regularizer=l2(REGULARIZER), strides=(1,1), max_pooling=True, 
            activation="relu", name=None): 

        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, 
            activation=activation,kernel_regularizer=kernel_regularizer)(x)
        if (max_pooling):
            x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
 
        return x

    def output_layer(self,x, classes):
        x = Dense(units=classes,  kernel_regularizer=l2(0))(x)
        x = Activation('softmax')(x)
        return x
    
    def dense_layer(self,x,units):
        x = Dense(units)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        return x

    def model(self):    
        # 1st LAYE
        x =  self.conv_layer(self.init, filters=96, kernel_size=(11,11), strides=(4,4),
            padding="valid", max_pooling=True, activation='relu', name='conv_1')

        x = BatchNormalization()(x) # apply batch normalisation.

        # 2nd Layer
        x =  self.conv_layer(x, filters=256, kernel_size=(5,5),strides=(1,1),
            padding="same", max_pooling=True, name="conv_2")

        x = BatchNormalization()(x) # apply batch normalisation.
        
        # 3RD LAYER
        x =  self.conv_layer(x, filters=384, kernel_size=(3,3),strides=(1,1),
            padding="same",max_pooling=False, name="conv_3")

        x = BatchNormalization()(x) # apply batch normalisation.


        # 4Th LAYER
        x =  self.conv_layer(x, filters=384, kernel_size=(3,3),strides=(1,1), 
            padding="same", max_pooling=False, name="conv_4")

        x = BatchNormalization()(x) # apply batch normalisation.


        # 5Th LAYER
        x =  self.conv_layer(x, filters=256, kernel_size=(3,3),strides=(1,1),
            padding="same", max_pooling=True, name="conv_5")

        x = BatchNormalization()(x) # apply batch normalisation.

        # 6 FLATTEN 
        x = Flatten()(x)

        # Fully Connected LAYER 1
        x = Dense(4096,  kernel_regularizer=l2(REGULARIZER))(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # FULLY CONNECTED LAYER 2
        x = Dense(4096,  kernel_regularizer=l2(REGULARIZER))(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # FULLY CONNECTED LAYER 2
        # x = Dense(1000,  kernel_regularizer=l2(0))(x)
        # x = Activation('relu')(x)
        # x = Dropout(0.5)(x)

        # Ouput Layer. Set class 
        output = self.output_layer(x, self.classes)

        model = Model(self.init, output, name='AlexNet')

        return model