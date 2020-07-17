from __future__ import division, print_function, absolute_import
import os
import csv
import cv2
import json
import random
import numpy as np
from PIL import Image
import pickle
import warnings
from urllib.parse import urlparse
from urllib import request
from io import BytesIO
import matplotlib.pyplot as plt
from keras.preprocessing import image
import itertools
import random
random.seed(1000)
np.random.seed(1000)
_EPSILON = 1e-8

def to_categorical(y, nb_classes=None):
    if nb_classes:
        y = np.asarray(y, dtype='int32')
        if len(y.shape) > 2:
            print("Warning: data array ndim > 2")
        if len(y.shape) > 1:
            y = y.reshape(-1)
        Y = np.zeros((len(y), nb_classes))
        Y[np.arange(len(y)), y] = 1.
        return Y
    else:
        y = np.array(y)
        return (y[:, None] == np.unique(y)).astype(np.float32)

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]
    
def load_image(in_image):
    # load image
    img = image.img_to_array(image.load_img(in_image, target_size=(224, 224)))
    # img = Image.open(in_image)
    # import pdb; pdb.set_trace()
    # img = cv2.imread(in_image)
    return img

def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

def convert_color(in_image, mode):
    return in_image.convert(mode)

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")
    
def build_image_dataset_from_dir(directory,
                                 dataset_file="my_tflearn_dataset.pkl",
                                 resize=None, convert_to_color=None,
                                 filetypes=None, shuffle_data=False,
                                 categorical_Y=False):
    try:
        X, Y = pickle.load(open(dataset_file, 'rb'))
    except Exception:
        X, Y = image_dirs_to_samples(directory, resize, convert_to_color, filetypes)
        if categorical_Y:
            Y = to_categorical(Y, np.max(Y) + 1) # First class is '0'
        if shuffle_data:
            X, Y = shuffle(X, Y)
        # pickle.dump((X, Y), open(dataset_file, 'wb'), protocol=4)
    return X, Y

def image_dirs_to_samples(directory, resize=None, convert_to_color=False,
                          filetypes=None):
    print("Starting to parse images...")
    if filetypes:
        if filetypes not in [list, tuple]: filetypes = list(filetypes)
    samples, targets = directory_to_samples(directory, flags=filetypes)
    for i, s in enumerate(samples):
        samples[i] = load_image(s)
        if convert_to_color:
            samples[i] = convert_color(samples[i],'RGB')

        # samples[i] = pil_to_nparray(samples[i])

        # if resize:
            # samples[i] = resize_image(samples[i], resize[0], resize[1])
            # samples[i] = random_crop(samples[i], resize)
        # samples[i] /= 255
    print("Parsing Done!")
    return samples, targets

def shuffle(*arrs):
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)

def directory_to_samples(directory, flags=None, filter_channel=False):
    samples = []
    targets = []
    label = 0
    classes = sorted(os.walk(directory).__next__()[1])
    for c in classes:
        c_dir = os.path.join(directory, c)
        try: # Python 2
            walk = os.walk(c_dir).next()
        except Exception: # Python 3
            walk = os.walk(c_dir).__next__()
        for sample in walk[2]:
            if not flags or any(flag in sample for flag in flags):
                if filter_channel:
                    if get_img_channel(os.path.join(c_dir, sample)) != 3:
                        continue
                samples.append(os.path.join(c_dir, sample))
                targets.append(label)
        label += 1
    return samples, targets


def get_img_channel(image_path):
    img = load_image(image_path)
    img = pil_to_nparray(img)
    try:
        channel = img.shape[2]
    except:
        channel = 1
    return channel

def onehot_to_cat(y):
    return np.argmax(y, axis=1)

def store_model(model, path,filename):
    # json_model = model.to_json()
    # with open(path+filename+".json", 'w') as file:
    #     file.write(json_model)
    model.save(path+filename+".h5")

def get_labels(y_onehot):
    y = onehot_to_cat(y_onehot)
    labels = np.empty(len(y), dtype=object)
    labels[y == 0 ] = "akiec"
    labels[y == 1 ] = "bcc"
    labels[y == 2 ] = "bkl"
    labels[y == 3 ] = "df"
    labels[y == 4 ] = "mel"
    labels[y == 5 ] = "nv"
    labels[y == 6 ] = "nv"

    return labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    labels = np.empty(len(classes), dtype=object)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    labels[tick_marks == 0 ] = "akiec"
    labels[tick_marks == 1 ] = "bcc"
    labels[tick_marks == 2 ] = "bkl"
    labels[tick_marks == 3 ] = "df"
    labels[tick_marks == 4 ] = "mel"
    labels[tick_marks == 5 ] = "nv"
    labels[tick_marks == 6 ] = "vasc"

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def write_csv_file(file, data, headers):
    if not (os.path.exists(file)):
        #write to file with headers
        with open(file,'wt',newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(h for h in headers)
            writer.writerow(data)
    else:
        with open(file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)

            import matplotlib.pyplot as plt

            
def plot_accuracy_loss_graph(path=None, name=None):
    # Plot training & validation accuracy values
    plt.style.use('ggplot')
    if name== None:
        name =""
    with open(path) as history_file:
        data = json.load(history_file)
        plt.plot(data['acc'])
        plt.plot(data['val_acc'])
        plt.plot(data['loss'])
        plt.plot(data['val_loss'])
        plt.title('Model Accuracy ('+name+')')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val', 'Loss', 'Val_loss'], loc='upper left')
        plt.show()
