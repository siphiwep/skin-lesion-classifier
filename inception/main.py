import sys
sys.path.append("..") 
from data_utils import *
import tensorflow as tf
from inception.inception_v3 import InceptionV3
from utils.transfer import set_non_trainable
from datetime import datetime
from utils.model_utils import ModelUtils
# from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D
# from keras.models import Model

   # # this is the model we will train
ACTIVATION= 'tanh' #relu or Mish

FOLDER = '2017'

if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 

    # # this is the model we will train
    model = InceptionV3(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=(224, 224, 3)), classes=3, activation=ACTIVATION)
    model = set_non_trainable(model)
    x = model.output
     # x = Dropout(0.2) (x)
    x =  tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(2048,activation=ACTIVATION)(x) 
    x=tf.keras.layers.Dense(2048,activation=ACTIVATION)(x) 
    x=tf.keras.layers.Dense(3,activation='softmax')(x) 
    model = tf.keras.Model(model.input, x, name='inceptionV3')
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
    util.confusion_matrix(name=model.name+'-ISIC-'+FOLDER+'-'+ACTIVATION)
    util.plot_loss_accuracy(path=model.name+'.json', name='InceptionV3-ISIC-'+FOLDER+' '+model.name)
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))