# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1E-O7ASKCG-N9FKgFhfQyekxOn-n-FACD
"""

"""# Imports"""

import numpy as np

from PIL import Image
from pathlib import Path

# Keras libraries and packages
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# * Intialize CNN
print('Intialize CNN')
classifier = Sequential()

# * 1. Convolution
print('Adding convolution layer')
classifier.add(Convolution2D(32, 4, strides=1, input_shape=(64, 64, 3), activation='relu'))

# * 2. Pooling
print('Adding pooling layer')
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
classifier.add(Dropout(0.25))

# * 3. Convolution
print('Adding convolution layer')
classifier.add(Convolution2D(64, 3, strides=1, activation='relu'))

# * 4. Pooling
print('Adding pooling layer')
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
classifier.add(Dropout(0.25))

# * 5. Flattening
print('Adding flattening layer')
classifier.add(Flatten())

# * 6. Full connection
print('Adding fully connected layer')
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))

# * Compile the CNN
print('Compiling the network')
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# * Generate more data by transforming existing images
print('Generating dataset')
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

parent_dir = '/content/drive/My Drive/Colab Notebooks/Signature Verification/'
training_set_path = parent_dir + 'dataset/training_set'
validation_set_path = parent_dir + 'dataset/val_set'

training_set = train_datagen.flow_from_directory(
        training_set_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
    
validation_set = validation_datagen.flow_from_directory(
        validation_set_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# * Save best weights while training
checkpoint = ModelCheckpoint(
        parent_dir + 'model/model.best.h5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True)
callbacks_list = [checkpoint]

# * Training the network
print('Training the network')
classifier.fit_generator(
        training_set,
        steps_per_epoch=500,
        epochs=15,
        validation_data=validation_set,
        validation_steps=300,
        callbacks=callbacks_list,
        shuffle=True)

# * Save model
print('Saving model')
classifier_json = classifier.to_json()
with open(parent_dir + 'model/model.json', 'w') as f:
    f.write(classifier_json)
classifier.save_weights(parent_dir + 'model/model.weights.h5')

# * Test the network
print('Predicting')
# test_image_path = parent_dir + 'dataset/test_set/forge/00301002.png' # Forged
# test_image_path = parent_dir + 'dataset/test_set/real/04601046.png' # Real
test_image_path = parent_dir + 'dataset/IMG_7989.jpg'

test_image = image.load_img(test_image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)

if result[0][0] >= 0.5:
    predicted = 'Real'
else:
    predicted = 'Forged'

# * Load model
print('Loading model')
model_path = parent_dir + 'model/model.best.h5'
classifier.load_weights(model_path)