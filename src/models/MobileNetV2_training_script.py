import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import os, h5py
import numpy as np

# training data
batch_size = 32
num_classes = 1
epochs = 50

# loading pretrained_model on imagenet with global average pooling and original sized convolutional filters
pretrained_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(160, 160, 3), alpha=1.0, include_top=False, weights='imagenet', pooling='avg')

# model topology
base = pretrained_model
base.trainable = False
model = Sequential()
model.add(base)
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.summary()

# optimizer settings
opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# set training generator
X_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255,
        preprocessing_function=None,
        data_format='channels_last',
        validation_split=0.0)

# set validation generator
y_datagen = ImageDataGenerator(rescale=1./255)

# get data for training
traingen = X_datagen.flow_from_directory(
                '../../data/external/cats_and_dogs_filtered/train/',
                target_size=(160, 160),
                batch_size=batch_size,
                class_mode='binary')

# get data for testing
testgen = y_datagen.flow_from_directory(
		'../../data/external/cats_and_dogs_filtered/validation/',
		target_size=(160, 160),
		batch_size=batch_size,
		class_mode='binary')

# callbacks for training
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
es = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, mode='auto')
mc = ModelCheckpoint('MobileNetV2_trained_classifier.hd5', monitor='val_loss', save_best_only=True)

# fit the model using the data from the generators and the callbacks above defined, with shuffling
model.fit_generator(traingen, epochs=epochs, shuffle=True, callbacks=[rlr, mc, es], validation_data=testgen)

# trained model score
scores = model.evaluate(testgen, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])