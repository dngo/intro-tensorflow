#!wget --no-check-certificate \
#    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
#    -O "/home/david/Desktop/happy-or-sad.zip"

import tensorflow as tf
import os
import zipfile

DESIRED_ACCURACY = 0.98

zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/home/david/Desktop/h-or-s")
zip_ref.close()

# Directory with our training happy pictures
train_h_dir = os.path.join('/home/david/Desktop/h-or-s/happy')

# Directory with our training sad pictures
train_s_dir = os.path.join('/home/david/Desktop/h-or-s/sad')

train_h_names = os.listdir(train_h_dir)
print(train_h_names[:10])

train_s_names = os.listdir(train_s_dir)
print(train_s_names[:10])

print('total training happy images:', len(os.listdir(train_h_dir)))
print('total training sad images:', len(os.listdir(train_s_dir)))

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc') > DESIRED_ACCURACY):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        "/home/david/Desktop/h-or-s",  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=2,
      epochs=20,
      verbose=1,
      callbacks=[callbacks])

import numpy as np
from tensorflow.keras.preprocessing import image

# files = [
#   '/home/david/Desktop/horse1.jpeg',
#   '/home/david/Desktop/horse2.jpeg',
#   '/home/david/Desktop/david.jpg',
#   '/home/david/Desktop/human01-01.png',
#   '/home/david/Desktop/smiley.png',
# ]

from pathlib import Path

pathlist = Path('/home/david/Desktop/dataset').glob('**/*')
for path_obj in pathlist:
  path = str(path_obj)
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=4)
  print(classes[0])
  print("---")
  if classes[0]>0.5:
    print(path + " is happy")
  else:
    print(path + " is sad")
