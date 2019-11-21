import sys
sys.path.append("..") 
from data_utils import *
from alexnet import AlexNet
from datetime import datetime
from utils.model_utils import ModelUtils
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16

if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 

    # # this is the model we will train
    alexnet = AlexNet(input_shape=(224, 224, 3), classes=3)
    model = alexnet.model()
    model.summary()
    
    util = ModelUtils(epochs=50)
    util.get_train_data(resize=(224,224))

    util.train(model)
    util.evaluate()
    util.save()
    util.confusion_matrix(title="AlexNet")
    util.plot_loss_accuracy(path=model.name+'.json', name="AlexNet")
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))