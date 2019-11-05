import sys
sys.path.append("..") 
from data_utils import *
from alexnet import AlexNet
from datetime import datetime
from utils.model_utils import ModelUtils
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16


DATASET_PATH = '../data/train/'
TEST_PATH = 'D:\Data/test/'
TEST_PATH_NAME=os.path.join(TEST_PATH, 'china.pkl')
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')

if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 

    # # this is the model we will train
    alexnet = AlexNet(input_shape=(227, 227, 3), classes=3)
    model = alexnet.model()
    model.summary()
    util = ModelUtils(epochs=50)
    util.get_train_data(resize=(227,227))
    # # util.get_test_data(resize=(227,227))
    util.train(model)
    util.evaluate()
    # # util.save()
    util.confusion_matrix()
    util.plot_loss_accuracy()
    
    # time_elapsed = datetime.now() - start 
    # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))