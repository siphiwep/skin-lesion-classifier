import sys
sys.path.append("..") 
from data_utils import *
from utils.transfer import set_non_trainable
from vgg.vgg19 import VGG19
from datetime import datetime
from utils.model_utils import ModelUtils
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D
from keras.models import Model

ACTIVATION='relu'
if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 

    # # this is the model we will train
    resnet50 = VGG19(input_shape=(224, 224, 3), classes=3, activation=ACTIVATION, include_top=False, weights='imagenet')
    model = resnet50.model()
    model = set_non_trainable(model)
    x = model.output
    x=Dense(1024,activation=ACTIVATION)(x) 
    model = Model(model.input, x, name='vgg19')
    model.summary()
    # util = ModelUtils(epochs=50)
    # util.get_train_data(resize=(224,224))

    # util.train(model)
    # util.evaluate()
    # util.save()
    # util.confusion_matrix(title="AlexNet")
    # util.plot_loss_accuracy(path=model.name+'.json', name="AlexNet")
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))