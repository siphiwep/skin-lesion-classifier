import sys
sys.path.append("..") 
from data_utils import *
from resnet.resnet50 import ResNet50
from utils.transfer import set_non_trainable
from datetime import datetime
from utils.model_utils import ModelUtils
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Model

ACTIVATION='relu' #relu or Mish

if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 

    # # this is the model we will train
    resnet50 = ResNet50(input_shape=(224, 224, 3), classes=7, activation=ACTIVATION, include_top=False, weights='imagenet')
    model = resnet50.model()
    model = set_non_trainable(model)
    x = model.output
    x =  AveragePooling2D(pool_size=(7,7))(x)
    x =  Flatten()(x)
    x=Dense(256,activation=ACTIVATION)(x) 
    # x=Dense(4096,activation=ACTIVATION)(x)
    x = Dropout(0.5) (x)
    x=Dense(7,activation='softmax')(x) 
    model = Model(model.input, x, name='resnet50')
    # model.summary()
    util = ModelUtils(epochs=60)
    util.get_train_data()
    util.train(model, name=ACTIVATION)
    util.evaluate()
    util.save(name=ACTIVATION)
    util.confusion_matrix(title=model.name)
    util.plot_loss_accuracy(path=model.name+'.json', name=model.name)
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))