import os
import cv2 as openCv
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from PIL import Image
from albumentations import Compose, CLAHE, CenterCrop, ToFloat
from keras.applications.imagenet_utils import preprocess_input

IMAGE_SIZE = 256

CROP_SIZE = 224


SEED = 1000

AUG_PATH='data' # Store the transformed image into the project folder
IMAGE_PATH="G:\SORTED-2017" #  Folder containing all the image to augment.
BINARY_DATA='G:\Data\mask'

def read_images(filepath):
    all_images = []
    images = [i for i in os.listdir(os.path.join(filepath)) if i.endswith('.jpg')]
    for path in images:
        all_images.append(openCv.imread(os.path.join(filepath, path)))
    return all_images

def resize_images(images, width=256, height=256):
    resized_images = []
    for image in images:
        resize_image = openCv.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
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
        openCv.imwrite(filename, image, [int(openCv.IMWRITE_JPEG_QUALITY), 90])
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

        if(parentdir == "train"): 

            for subdir in os.listdir(os.path.join(IMAGE_PATH, parentdir)):
                print("Reading sub-folders in {0} ".format(subdir))
                for mmetadir in os.listdir(os.path.join(IMAGE_PATH, parentdir, subdir)):
                    print("Reading sub-folders in {0} ".format(mmetadir))
                    images = read_images(os.path.join(IMAGE_PATH, parentdir, subdir, mmetadir))
                    # masks = read_images(os.path.join(BINARY_DATA, parentdir, subdir))
                    # no_hair_images = removeHair(images)
                    # segmented = segment_images(no_hair_images,masks)
                    # print("{} will be rotated and flipped".format(len(images)))
                    resized_images = resize_images(images)
                    rotated_images = rotate_images(resized_images)
                    # print("Rotated {}".format(len(cropped_images_rot)))
                    # save_images(filepath='/'.join([AUG_PATH, 'train', parentdir, subdir]), images=cropped_images_rot, prefix="rotated")

                    # flipped_images = flip_images(images)
                    cropped_images = random_crop(rotated_images, 3)
                    # import pdb; pdb.set_trace()
                    # print("Flipped  {}".format(len(flipped_images)))
                    # im = applyClahe(images)
                    # print("Cropped  {}".format(len(cropped_images)))
                    # flipped_rotated =  np.concatenate((rotated_images, flipped_images))
                    # cropped_images = random_crop(flipped_rotated,5)
                    save_images(filepath='/'.join([AUG_PATH, parentdir, subdir,mmetadir]), images=cropped_images, prefix="im")

        else:
            for subdir in os.listdir(os.path.join(IMAGE_PATH, parentdir)):
                    print("Reading sub-folders in {0} ".format(subdir))
                    for mmetadir in os.listdir(os.path.join(IMAGE_PATH, parentdir, subdir)):
                        print("Reading sub-folders in {0} ".format(mmetadir))
                        images = read_images(os.path.join(IMAGE_PATH, parentdir, subdir, mmetadir))
                        # masks = read_images(os.path.join(BINARY_DATA, parentdir, subdir))
                        # no_hair_images = removeHair(images)
                        # segmented = segment_images(no_hair_images,masks)
                        # print("{} will be rotated and flipped".format(len(images)))
                        resized_images = resize_images(images)
                        # print("Rotated {}".format(len(cropped_images_rot)))
                        # save_images(filepath='/'.join([AUG_PATH, 'train', parentdir, subdir]), images=cropped_images_rot, prefix="rotated")

                        # flipped_images = flip_images(images)
                        cropped_images = random_crop(resized_images, 1)

                        save_images(filepath='/'.join([AUG_PATH, parentdir, subdir,mmetadir]), images=cropped_images, prefix="im")

def removeHair(images):
    r_images = []
    for img in images:
        grey = openCv.cvtColor(img, openCv.COLOR_RGB2GRAY)
        kernel = openCv.getStructuringElement(1,(17,17))
        blackhat = openCv.morphologyEx(grey, openCv.MORPH_BLACKHAT, kernel)
        ret,thresh2 = openCv.threshold(blackhat,10,255,openCv.THRESH_BINARY)
        dst = openCv.inpaint(img,thresh2,1,openCv.INPAINT_TELEA)
        color = openCv.cvtColor(img, openCv.COLOR_RGB2GRAY)
        r_images.append(dst)
    return r_images

def segment_images(images, masks):
    segmented=[]
    for image, mask in zip(images, masks):
        mask_out = openCv.subtract(mask,image)
        mask_out = openCv.subtract(mask,mask_out)
        segmented.append(mask_out)
    return segmented

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
    
                
                
