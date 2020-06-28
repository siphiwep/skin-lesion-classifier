import os
import numpy as np
import json
import keras as ke
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from data_utils import build_image_dataset_from_dir, get_labels, onehot_to_cat, plot_confusion_matrix, plot_accuracy_loss_graph
from keras.applications.vgg19 import preprocess_input
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import random
tf.set_random_seed(1000)
random.seed(1000)
np.random.seed(1000)

FOLDER = '2017'
class ModelUtils():

    def __init__(self, epochs=2,test_split=0.10, validation_split=0.20):
        self.epochs=epochs
        self.test_split=test_split
        self.validation=validation_split
        self.batch_size = 32

    def get_train_data(self, name=FOLDER, folder='../data/train', resize=None):
        self.x, self.y = build_image_dataset_from_dir(os.path.join(folder, name),
            dataset_file=os.path.join(folder, name+'.pkl'),
            resize=resize,
            filetypes=['.jpg'],
            convert_to_color=False,
            shuffle_data=True,
            categorical_Y=True)
        
        # self.trainX, self.valX, self.trainY, self.valY = train_test_split(self.x, self.y, test_size=self.validation, random_state=1000)
        # self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.trainX, self.trainY, test_size=self.test_split, random_state=1000)

        print("Training on {0}".format(len(self.x)))
        # print("Validating on {0}".format(len(self.valX)))
        # print("Testing on {0}".format(len(self.testX)))
    
    def get_test_data(self, name=FOLDER, folder='../data/test', resize=None):
        self.testX, self.testY = build_image_dataset_from_dir(os.path.join(folder, name),
            dataset_file=os.path.join(folder, name+'.pkl'),
            resize=resize,
            filetypes=['.jpg'],
            convert_to_color=False,
            shuffle_data=True,
            categorical_Y=True)

        print("Testing on {0} ".format(len(self.testX)))

    def get_val_data(self, name=FOLDER, folder='../data/val', resize=None):
        self.valX, self.valY = build_image_dataset_from_dir(os.path.join(folder, name),
            dataset_file=os.path.join(folder, name+'.pkl'),
            resize=resize,
            filetypes=['.jpg'],
            convert_to_color=False,
            shuffle_data=True,
            categorical_Y=True)

        print("Validating on {0} ".format(len(self.valX)))
        
    
    def mean_subtraction(self):
        mean = np.mean(self.x, axis=0)
        self.x -= mean
        self.testX -= mean
        self.valX -= mean
        

    def train(self, model, name=None):
        self.model = model
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer(), 
            metrics=['accuracy'])

        aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            # rotation_range=90, 
			zoom_range=0.15,
			width_shift_range=0.2,
			height_shift_range=0.2,
			# shear_range=0.25,
			# horizontal_flip=True,
            # vertical_flip=True,
			fill_mode="nearest"
        )
        valAug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            # rotation_range=90, 
			# zoom_range=0.15,
			# width_shift_range=0.2,
			# height_shift_range=0.2,
			# shear_range=0.25,
			# horizontal_flip=True,
            # vertical_flip=True,
			fill_mode="nearest"
        )
        if(K.image_dim_ordering() == 'th'):
            self.x = np.moveaxis(self.x, -1, 1)
            self.valX = np.moveaxis(self.valX, -1, 1)
        if(os.path.exists('../models/'+self.model.name+FOLDER+name+'.h5')):
            self.model.load_weights('../models/'+self.model.name+FOLDER+name+'.h5') 
        else:
       
            self.history = self.model.fit_generator(aug.flow(self.x,self.y, batch_size=self.batch_size, shuffle=True, seed=1000),
                steps_per_epoch=len(self.x)/self.batch_size ,epochs=self.epochs, verbose=1, 
                validation_steps=len(self.valX) / self.batch_size,
                validation_data=valAug.flow(self.valX, self.valY, batch_size=self.batch_size, shuffle=True, seed=1000))

            with open(self.model.name+'.json', 'w') as file:
                json.dump(self.history.history, file)


    def evaluate(self):
        score = self.model.evaluate(self.testX, self.testY)
      
        print(score)
        print("%s: %.2f%%" % (self.model.metrics_names[-1], score[-1]))

    def save(self, folder='../models', name=None):
        self.model.save_weights(folder+'/'+self.model.name+FOLDER+name+'.h5')

    def optimizer(self):
        return SGD(lr=0.00001, momentum=0.9, decay=0.0005)

    def confusion_matrix(self, title=None):
        if title== None:
            title = self.model.name+FOLDER
        predictions = self.model.predict(self.testX)
        class_counts = onehot_to_cat(self.testY)
        print("Class counts")
        print(np.unique(class_counts, return_counts=True))
        print("===========================================0======")
        print("F MEASURE")

        labels = list(set(get_labels(self.testY)))
        # print(classification_report())
        cm = confusion_matrix(get_labels(self.testY),get_labels(predictions))
        print("Confusion Matrix {}".format(cm))
        plot_confusion_matrix(cm, labels, title=title)

    def plot_loss_accuracy(self, path, name):
        plot_accuracy_loss_graph(path, name)