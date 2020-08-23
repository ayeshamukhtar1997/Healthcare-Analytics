import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense
from keras import backend as K
import numpy as np
from keras.preprocessing.image import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import sys



width, height = 128,128
train_dir = ""   # Add train directory
validate_dir = ""  # Add test Directory
train_sample = 42000
valid_samples = 7000
epochs = 1
batch_s = 256

if K.image_data_format() == 'channels_first':
    shape = (3,width,height)
else:
    shape = (width,height,3)
    
    
train_dg = ImageDataGenerator(
rescale = 1./255, zoom_range =0.2)

test_dg = train_dg = ImageDataGenerator(
rescale = 1./255)

tg = train_dg.flow_from_directory(
train_dir, target_size =(width,height), batch_size = batch_s, class_mode = 'categorical'

)

vg = test_dg.flow_from_directory(
validate_dir, target_size =(width,height), batch_size = batch_s, class_mode = 'categorical'

)

print("CNN Model 1")
def model_1():
	model = Sequential([
		add(Conv2D(64, (3,3),strides = (1,1), input_shape = shape,kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(MaxPooling2D(pool_size=(2, 2), strides= (2,2))),

		add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(MaxPooling2D(pool_size=(2, 2), strides= (2,2))),

		add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(MaxPooling2D(pool_size=(2, 2), strides= (2,2))), add(Flatten()),
		add(Dense(2048)), add(keras.layers.ELU()), add(BatchNormalization()),

		add(Dropout(0.5)), add(Dense(7, activation='softmax'))])

		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model


def model_2():
	model = Sequential([
		add(Conv2D(64, (3,3),strides = (1,1), input_shape = shape,kernel_initializer='glorot_uniform')),
		
		add(MaxPooling2D(pool_size=(2, 2), strides= (2,2))),

		add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),

		add(MaxPooling2D(pool_size=(2, 2), strides= (2,2))),

		add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),

		add(MaxPooling2D(pool_size=(2, 2), strides= (2,2))), add(Flatten()),
		add(Dense(2048)), 

		add(Dropout(0.5)), add(Dense(7, activation='sigmoid'))])

		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

def model_3():
	model = Sequential([
		add(Conv2D(64, (3,3),strides = (1,1), input_shape = shape,kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(MaxPooling2D(pool_size=(2, 2), strides= (2,2))),

		add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform')),
		add(keras.layers.ELU()), add(BatchNormalization()),

		add(MaxPooling2D(pool_size=(2, 2), strides= (2,2))), add(Flatten()),
		add(Dense(2048)), add(keras.layers.ELU()), add(BatchNormalization()),

		add(Dropout(0.5)), add(Dense(7, activation='softmax'))])

		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

def model_4():
	model = Sequential([
	add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = shape)),
	add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',  activation ='relu')),
	add(MaxPooling2D(pool_size=(2,2))),
	add(Dropout(0.25)),
	add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu')),
	add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu')),
	add(MaxPooling2D(pool_size=(2,2), strides=(2,2))),
	add(Dropout(0.25)),
	add(Flatten()),
	add(Dense(256, activation = "relu")),
	add(Dropout(0.5)),
	add(Dense(7, activation = "softmax"))])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


#Run each model function and save the model by changing the name
model = model_1()
model.summary()
model = model.fit_generator(
tg, steps_per_epoch = train_sample//batch_s,
    epochs =epochs,
    validation_data = vg,
    validation_steps = valid_samples // batch_s
)
model.save('/content/Train/model_1.h5')


import matplotlib.pyplot as plt
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'])
plt.show()


plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()





