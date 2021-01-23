import sys 
sys.path.append("..") 
from data_utils import *
from utils.transfer import set_non_trainable
from xception.xception_v1 import Xception
from datetime import datetime
from utils.model_utils import ModelUtils
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D
from keras.models import Model
# valid values: Mish, relu, selu, tanh
ACTIVATION='relu'

FOLDER = '2017'
if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 

    # # this is the model we will train
    xception = Xception(input_shape=(224, 224, 3), classes=3, activation=ACTIVATION, include_top=False, weights='imagenet')
    model = xception.model()
    model = set_non_trainable(model)
    x = model.output
    # x = Dropout(0.2) (x)
    x=Dense(2048,activation=ACTIVATION)(x)
    x=Dense(2048,activation=ACTIVATION)(x) 
    # x = Dropout(0.2) (x)

    x=Dense(3,activation='softmax')(x) 
    model = Model(model.input, x, name='xception')
    model.summary()
    

    util = ModelUtils(epochs=60)
    if FOLDER == '2017':
        util.get_train_data(name='',folder='../data/'+FOLDER+'/train')
        util.get_val_data(name='', folder='../data/'+FOLDER+'/val')
        util.get_test_data(name='', folder='../data/'+FOLDER+'/test')
    else:
        util.get_train_data()

    util.train(model, name=ACTIVATION)
    util.evaluate()
    util.save(name=ACTIVATION)
    util.confusion_matrix(name=model.name)
    util.plot_loss_accuracy(path=model.name+'.json', name=model.name)
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))