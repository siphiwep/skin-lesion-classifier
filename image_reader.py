from keras.preprocessing.image import ImageDataGenerator

def getTrainGenerator():
    return ImageDataGenerator (
        rotation_range=25,
	    zoom_range=0.1,
	    width_shift_range=0.1,
	    height_shift_range=0.1,
	    shear_range=0.2,
	    horizontal_flip=True,
	    fill_mode="nearest")

def getValidationGenerator():
    return ImageDataGenerator()

def getTestGenerator():
    return ImageDataGenerator()