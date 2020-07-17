import os
import numpy as np
import json
import keras as ke
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score
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

FOLDER = '2018'
class ModelUtils():

    def __init__(self, epochs=2,test_split=0.10, validation_split=0.20):
        self.epochs=epochs
        self.test_split=test_split
        self.validation=validation_split
        self.batch_size = 64

    def get_train_data(self, name=FOLDER, folder='../data/', resize=None):
        self.x, self.y = build_image_dataset_from_dir(os.path.join(folder, name),
            dataset_file=os.path.join(folder, name+'.pkl'),
            resize=resize,
            filetypes=['.jpg'],
            convert_to_color=False,
            shuffle_data=True,
            categorical_Y=True)
        
        self.trainX, self.valX, self.trainY, self.valY = train_test_split(self.x, self.y, test_size=self.validation, random_state=1000)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.trainX, self.trainY, test_size=self.test_split, random_state=1000)

        print("Training on {0}".format(len(self.trainX)))
        print("Validating on {0}".format(len(self.valX)))
        print("Testing on {0}".format(len(self.testX)))
    
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
            # preprocessing_function=preprocess_input,
            rotation_range=90, 
			# zoom_range=0.15,
			width_shift_range=0.2,
			# height_shift_range=0.2,
			shear_range=0.25,
			horizontal_flip=True,
            # vertical_flip=True,
			fill_mode="nearest"
        )
        valAug = ImageDataGenerator(
            # preprocessing_function=preprocess_input,
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

            with open(self.model.name+FOLDER+name+'.json', 'w') as file:
                json.dump(self.history.history, file)


    def evaluate(self):
        score = self.model.evaluate(self.testX, self.testY)
      
        print(score)
        print("%s: %.2f%%" % (self.model.metrics_names[-1], score[-1]))

    def save(self, folder='../models', name=None):
        self.model.save_weights(folder+'/'+self.model.name+FOLDER+name+'.h5')

    def optimizer(self):
        return SGD(lr=0.001, momentum=0.9, decay=0.0005)

    def confusion_matrix(self, name=None):
        predictions = self.model.predict(self.valX)
        labels = list(set(get_labels(self.valY))) 
        print(labels)
        target_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        print("Classification report for " + FOLDER + " ---> " +self.model.name)
        # print(precision_recall_fscore_support(np.argmax(predictions, axis=1), np.argmax(self.valY, axis=1)))
        print("F1 SCORE:")
        print(f1_score(np.argmax(self.valY, axis=1), np.argmax(predictions, axis=1), average=None))
        print("RECALL:")
        print(recall_score(np.argmax(self.valY, axis=1), np.argmax(predictions, axis=1), average=None))

        print("PRECISION:")
        print(precision_score(np.argmax(self.valY, axis=1), np.argmax(predictions, axis=1),average=None))

        print("SPECIFICITY:")
        # self.fpr, self.tpr, _ = roc_curve(np.argmax(self.valY, axis=1),predictions[:,1], )
        # self.auc = roc_auc_score(np.argmax(self.valY, axis=1), np.argmax(predictions, axis=1))
        print(classification_report(get_labels(self.valY), get_labels(predictions)))
        cm = confusion_matrix(get_labels(self.valY),get_labels(predictions))
        # tn, fp, fn, tp = confusion_matrix(get_labels(self.valY),get_labels(predictions)).ravel()
        # print("True Positive {} False Positive {} False Negative {} True Positive {}".format(tn, fp, fn, tp))
        # print("TN {}".format(cm[0][0]))
        # print("FP {}".format(cm[0][1]))
        # print("FN {}".format(cm[1][0]))
        # print("TP {}".format(cm[1][1]))
        # specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        # print(specificity)
        print("Confusion Matrix {}\n".format(cm))
        plot_confusion_matrix(cm, labels, title=name if not None else self.model.name+FOLDER)

    def plot_loss_accuracy(self, path, name):
        plot_accuracy_loss_graph(path, name)