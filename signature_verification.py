# -*- coding: utf-8 -*-
import argparse
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from PIL import Image

app = Flask(__name__)
CORS(app)

def train(dataset_dir, model_path):
	# * Intialize CNN
	print('Intialize CNN')
	classifier = Sequential()

	# Convolution
	print('Adding convolution layer')
	classifier.add(Convolution2D(32, 4, strides=1, input_shape=(64, 64, 3), activation='relu'))

	# Pooling
	print('Adding pooling layer')
	classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	classifier.add(Dropout(0.25))

	# Convolution
	print('Adding convolution layer')
	classifier.add(Convolution2D(64, 3, strides=1, activation='relu'))

	# Pooling
	print('Adding pooling layer')
	classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	classifier.add(Dropout(0.25))

	# Flattening
	print('Adding flattening layer')
	classifier.add(Flatten())

	# Full connection
	print('Adding fully connected layer')
	classifier.add(Dense(units=128, activation='relu'))
	classifier.add(Dropout(0.5))
	classifier.add(Dense(units=2, activation='softmax'))

	# Compile the CNN
	print('Compiling the network')
	classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Generate data by transforming existing images
	print('Generating dataset')
	train_datagen = ImageDataGenerator(
			rotation_range=15,
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)

	validation_datagen = ImageDataGenerator(rescale=1./255)

	training_set_path = Path(dataset_dir).absolute() / 'training_set'
	validation_set_path = Path(dataset_dir).absolute() / 'val_set'

	training_set = train_datagen.flow_from_directory(
			str(training_set_path),
			target_size=(64, 64),
			batch_size=32,
			class_mode='categorical')
		
	validation_set = validation_datagen.flow_from_directory(
			str(validation_set_path),
			target_size=(64, 64),
			batch_size=32,
			class_mode='categorical')

	# Save best weights while training
	checkpoint = ModelCheckpoint(
			str(Path(model_path).absolute() / 'model.best.h5'),
			monitor='val_acc',
			verbose=1,
			save_best_only=True)
	callbacks_list = [checkpoint]

	# Training the network
	print('Training the network')
	classifier.fit_generator(
			training_set,
			steps_per_epoch=500,
			epochs=15,
			validation_data=validation_set,
			validation_steps=300,
			callbacks=callbacks_list,
			shuffle=True)

def predict(test_image_path, load_model_path):
	print(f'Predicting image `{test_image_path}`')
	test_image = image.load_img(test_image_path, target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)

	print('Loading model')
	classifier = Sequential()
	classifier.load_weights(load_model_path)

	result = classifier.predict(test_image)

	if int(np.argmax(result[0])) == 1:
		print('Genuine Signature')
	else:
		print('Forged Signature')

def execute(training_dir=None, save_model_path=None, test_image_path=None, load_model_path=None):
	if training_dir is not None:
		if save_model_path is None:
			print('Provide a directory to save model.')
			return
		
		train(training_dir, save_model_path)
		return
	
	if test_image_path is not None:
		if load_model_path is None:
			print('Provide path to model to predict with.')
			return

		predict(test_image_path)
		return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, help='Directory in which the dataset lies.', default=None)
	parser.add_argument('-s', '--save', type=str, help='Directory in which model will be saved.', default=None)
	parser.add_argument('-t', '--test', type=str, help='Path of the image to test on.', default=None)
	parser.add_argument('-l', '--load', type=str, help='Path of the model to load.', default=None)

	args = parser.parse_args()

	training_dir = args.dir
	save_model_path = args.save
	test_image_path = args.test
	load_model_path = args.load

	execute(training_dir, save_model_path, test_image_path, load_model_path)