"""
    Author: Pike Msonda
    Description: ResNet implementation using Keras api
"""

from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D, merge, Reshape
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import layers


class ResNet:

    def __init__(self, input_shape, classes):
        self.input_shape = Input(input_shape)
        self.classes = classes


    def identity_layer(self, input_shape, filters, kernel_size, strides=(2,2)):
        filter1, filter2, filter3 = filters
        axis=3

        #import pdb; pdb.set_trace()

        x = Conv2D(filter1,(1,1), kernel_initializer='he_normal')(input_shape)
        x = BatchNormalization(axis=axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(filter2, kernel_size, padding='same',
            kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=axis)(x)
        x = Activation('relu')(x)


        x = Conv2D(filter3,(1,1), kernel_initializer='he_normal')(x)
        x =  BatchNormalization(axis=axis)(x)

        x = layers.add([x, input_shape])
        x = Activation('relu')(x)

        return x

    def conv_layer(self, input_shape, filters, kernel_size, strides=(2,2)):
        
        filter1, filter2, filter3 = filters
        axis=3

        x = Conv2D(filter1, (1, 1), strides=strides,
                      kernel_initializer='he_normal')(input_shape)
        x = BatchNormalization(axis=axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(filter2, kernel_size, padding='same',
                      kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=axis)(x)
        x = Activation('relu')(x)

        x = Conv2D(filter3, (1, 1), kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=axis)(x)

        shortcut =Conv2D(filter3,(1,1), strides=strides,
            kernel_initializer='he_normal')(input_shape)
        shortcut = BatchNormalization(axis=axis)(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x

    def model(self):

        axis=3
        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(self.input_shape)
        x = Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=axis)(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = self.conv_layer(x,[64, 64, 256], (3,3), strides=(1, 1))
        x = self.identity_layer(x,[64, 64, 256], (3,3))
        x = self.identity_layer(x,[64, 64, 256], (3,3))

        x = self.conv_layer(x,[128, 128, 512], (3,3))
        x = self.identity_layer(x,[128, 128, 512], (3,3))
        x = self.identity_layer(x, [128, 128, 512], (3,3))
        x = self.identity_layer(x, [128, 128, 512], (3,3))

        x = self.conv_layer(x, [256, 256, 1024],(3,3))
        x = self.identity_layer(x, [256, 256, 1024],(3,3))
        x = self.identity_layer(x, [256, 256, 1024],(3,3))
        x = self.identity_layer(x, [256, 256, 1024],(3,3))
        x = self.identity_layer(x, [256, 256, 1024],(3,3))
        x = self.identity_layer(x, [256, 256, 1024],(3,3))

        x = self.conv_layer(x, [512, 512, 2048],3)
        x = self.identity_layer(x, [512, 512, 2048],3)
        x = self.identity_layer(x, [512, 512, 2048],3)

        # x = Flatten()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.classes, activation='softmax')(x)

        model = Model(self.input_shape, x, name='ResNet')

        return model
