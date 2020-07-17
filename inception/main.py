import sys
sys.path.append("..") 
from data_utils import *
from inception.inception_v3 import InceptionV3
from utils.transfer import set_non_trainable
from datetime import datetime
from utils.model_utils import ModelUtils
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D
from keras.models import Model

ACTIVATION='relu'
if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 

    # # this is the model we will train
    inceptionV3 = InceptionV3(input_shape=(224, 224, 3), classes=7, activation=ACTIVATION, include_top=False, weights='imagenet')
    model = inceptionV3.model()
    model = set_non_trainable(model)
    x = model.output
    x=Dense(1024,activation=ACTIVATION)(x) 
    x=Dense(1024,activation=ACTIVATION)(x) 
    x=Dense(7,activation='softmax')(x) 
    model = Model(model.input, x, name='inceptionV3')
    # model.summary()
    util = ModelUtils(epochs=5)
    # util.get_train_data(resize=(224,224))
    util.get_train_data()
    # util.get_val_data()
    # util.get_test_data()
    util.train(model, name=ACTIVATION)
    util.evaluate()
    util.save(name=ACTIVATION)
    util.confusion_matrix(title=model.name)
    util.plot_loss_accuracy(path=model.name+'.json', name=model.name)
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))