import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

print('products training ' + str(len(os.listdir('/home/david/workspace/s3_assets/products/'))))
print('users training ' + str(len(os.listdir('/home/david/workspace/s3_assets/users/'))))

try:
    os.mkdir('/tmp/products-v-users')
    os.mkdir('/tmp/products-v-users/training')
    os.mkdir('/tmp/products-v-users/training/products')
    os.mkdir('/tmp/products-v-users/training/users')
    os.mkdir('/tmp/products-v-users/testing')
    os.mkdir('/tmp/products-v-users/testing/products')
    os.mkdir('/tmp/products-v-users/testing/users')
except OSError:
    pass

# takes a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over
#
# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
# random.sample(list, len(list)) shuffles a list
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

PRODUCT_SOURCE_DIR = "/home/david/workspace/s3_assets/products/"
TRAINING_PRODUCTS_DIR = "/tmp/products-v-users/training/products/"
TESTING_PRODUCTS_DIR =  "/tmp/products-v-users/testing/products/"

USER_SOURCE_DIR = "/home/david/workspace/s3_assets/users/"
TRAINING_USERS_DIR = "/tmp/products-v-users/training/users/"
TESTING_USERS_DIR =  "/tmp/products-v-users/testing/users/"

split_size = .9
split_data(PRODUCT_SOURCE_DIR, TRAINING_PRODUCTS_DIR, TESTING_PRODUCTS_DIR, split_size)
split_data(USER_SOURCE_DIR, TRAINING_USERS_DIR, TESTING_USERS_DIR, split_size)

print('products training ' + str(len(os.listdir('/tmp/products-v-users/training/products/'))))
print('users training ' + str(len(os.listdir('/tmp/products-v-users/training/users/'))))
print('products testing ' + str(len(os.listdir('/tmp/products-v-users/testing/products/'))))
print('users testing ' + str(len(os.listdir('/tmp/products-v-users/testing/users/'))))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR =  "/tmp/products-v-users/training"
train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


VALIDATION_DIR = "/tmp/products-v-users/testing/"
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = train_datagen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

history = model.fit_generator(train_generator,
                              epochs=20,
                              verbose=1,
                              validation_data=validation_generator)

model.save('abound.h5')
model = tf.keras.models.load_model('abound.h5')

import numpy as np
from tensorflow.keras.preprocessing import image

from pathlib import Path

pathlist = Path('/home/david/Desktop/dataset').glob('**/*')
for path_obj in pathlist:
  path = str(path_obj)
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(path + " is a user")
  else:
    print(path + " is a product")
