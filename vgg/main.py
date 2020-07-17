import sys 
sys.path.append("..") 
from data_utils import *
from utils.transfer import set_non_trainable
from vgg.vgg19 import VGG19
from datetime import datetime
from utils.model_utils import ModelUtils
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D
from keras.models import Model
# valid values: Mish, relu, selu, tanh
ACTIVATION='tanh'
if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 

    # # this is the model we will train
    vgg = VGG19(input_shape=(224, 224, 3), classes=7, activation=ACTIVATION, include_top=False, weights='imagenet')
    model = vgg.model()
    model = set_non_trainable(model)
    x = model.output
    # x = Dropout(0.2) (x)
    x=Dense(4096,activation=ACTIVATION)(x) 
    x=Dense(4096,activation=ACTIVATION)(x)
    # x = Dropout(0.2) (x)

    x=Dense(7,activation='softmax')(x) 
    model = Model(model.input, x, name='vgg19')
    # model.summary()

    util = ModelUtils(epochs=60)
    util.get_train_data()
    # util.get_val_data()
    # util.get_test_data()
    # util.mean_subtraction()
    util.train(model, name=ACTIVATION)
    util.evaluate()
    # util.save(name=ACTIVATION)
    util.confusion_matrix(name=model.name)
    util.plot_loss_accuracy(path=model.name+'.json', name=model.name)
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))