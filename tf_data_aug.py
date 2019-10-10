import os
import cv2 as openCv
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
from albumentations import Compose, CLAHE, CenterCrop, ToFloat
IMAGE_SIZE = 256

CROP_SIZE = 224

IMAGE_PATH = ''

SEED = 1000

AUG_PATH='data' # Store the transformed image into the project folder
IMAGE_PATH="C:/Users/Siphiwe Phiri/Documents/thesis/data" #  Folder containing all the image to augment.

def resize_images(filepath, width=256, height=256):
    resized_images = []
    images = [i for i in os.listdir(os.path.join(filepath)) if i.endswith('.jpg')]
    for image in images:
        img = openCv.imread(os.path.join(filepath, image))
        # imgClahe = applyClahe(img)
        resize_image = openCv.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
        resized_images.append(resize_image)
    np.array(resized_images, dtype ="float") / 255.0
    return resized_images
def apply_central_crop():
    return Compose([
        CenterCrop(CROP_SIZE, CROP_SIZE)
    ], p=1)
def central_crop(images):
    cropped_images = []
    croper = apply_central_crop()
    for image in images:
        out = croper(**{'image': image})
        cropped_images.append(out['image'])
    return cropped_images

def save_images(filepath, images, prefix="untitled"):
    for index, image in enumerate(images):
        filename = filepath+'/'+prefix+'_'+str(index)+'.jpg'
        openCv.imwrite(filename, image)
        # Image.fromarray(image, mode='RGB').save(filename)
        # import pdb; pdb.set_trace()
        # imageToSave = Image.fromarray(image)

def applyClahe(image):
    clahe = openCv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)  

def rotate_images(images):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in images:
            X_rotate.append(img) #append original image
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype =np.uint8)
    return X_rotate
    
def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    # tf_img1 = tf.image.flip_left_right(X)
    # tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.uint8)
    return X_flip

def random_crop(images, samples=2):
    x_random_crops = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE,3))
    
    tf_cache = tf.image.random_crop(X, [CROP_SIZE, CROP_SIZE,3], SEED)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in images:
            for _ in range(samples):
                random_cropped_image = sess.run(tf_cache, feed_dict = {X: img})
                x_random_crops.append(random_cropped_image)

    x_random_crops = np.array(x_random_crops, dtype= np.uint8)

    return x_random_crops

def add_augs():
    for parentdir in os.listdir(IMAGE_PATH):
        print("Reading sub-folders in {0} ".format(parentdir))
        for subdir in os.listdir(os.path.join(IMAGE_PATH, parentdir)):
            print("Reading sub-folders in {0} ".format(subdir))

            images = resize_images(os.path.join(IMAGE_PATH, parentdir, subdir))
            # print("{} will be rotated and flipped".format(len(images)))
            rotated_images = rotate_images(images)
            # print("Rotated {}".format(len(cropped_images_rot)))
            # save_images(filepath='/'.join([AUG_PATH, 'train', parentdir, subdir]), images=cropped_images_rot, prefix="rotated")

            # flipped_images = flip_images(images)
            cropped_images_fli = central_crop(rotated_images)
            # print("Flipped  {}".format(len(flipped_images)))
            # im = applyClahe(images)
            # print("Cropped  {}".format(len(cropped_images_fli)))
            # flipped_rotated =  np.concatenate((rotated_images, flipped_images))
            # cropped_images = random_crop(flipped_rotated,5)
            save_images(filepath='/'.join([AUG_PATH, 'train', parentdir, subdir]), images=cropped_images_fli, prefix="im")
        
def create_dataset():
     for parentdir in os.listdir(AUG_PATH):
        if(parentdir == 'all'):
            pass
        else:
            print("Reading sub-folders in {0} ".format(parentdir))
            for subdir in os.listdir(os.path.join(AUG_PATH, parentdir)):
                print("Reading sub-folders in {0} ".format(subdir))
                images =  read_images(folder=os.path.join(AUG_PATH, parentdir, subdir))
                cropped_images = random_crop(images)
                print("Cropped  {}".format(len(cropped_images)))
                save_images(filepath='/'.join([AUG_PATH, 'all', parentdir, subdir]), images=cropped_images, prefix="cropped")

if __name__ == "__main__":
    add_augs()
    # create_dataset()
    
                
                
