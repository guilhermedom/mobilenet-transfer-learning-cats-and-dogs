"""Transfer and train a MobileNetV2 on the Cats and Dogs dataset.
"""
import zipfile

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        ModelCheckpoint,
                                        EarlyStopping)


BATCH_SIZE = 32
NUM_CLASSES = 1
EPOCHS = 50

# Loading pretrained model on imagenet with global average pooling and
# original sized convolutional filters.
pretrained_model = MobileNetV2(input_shape=(160, 160, 3),
                               alpha=1.0,
                               include_top=False,
                               weights='imagenet',
                               pooling='avg')

# Model topology.
base = pretrained_model
base.trainable = False
model = Sequential()
model.add(base)
model.add(Dense(NUM_CLASSES))
model.add(Activation('sigmoid'))

model.summary()

# Optimizer settings.
opt = RMSprop(lr=0.001, decay=1e-6)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Set train data generator with data augmentation.
train_image_gen = ImageDataGenerator(
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
        validation_split=0.0
)

# Set validation data generator.
validation_image_gen = ImageDataGenerator(rescale=1./255)

# Extract dataset from the repository zipped folder.
zip_object = zipfile.ZipFile('../../data/raw/cats_and_dogs_filtered.zip')
zip_object.extractall('../../data/raw/')
zip_object.close()

# Data generator for training.
traingen = train_image_gen.flow_from_directory(
                '../../data/raw/cats_and_dogs_filtered/train/',
                target_size=(160, 160),
                batch_size=BATCH_SIZE,
                class_mode='binary')

# Data generator for validation.
validgen = validation_image_gen.flow_from_directory(
		'../../data/raw/cats_and_dogs_filtered/validation/',
		target_size=(160, 160),
		batch_size=BATCH_SIZE,
		class_mode='binary')

# Callbacks for training.
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
es = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, mode='auto')
mc = ModelCheckpoint('../../models/mobilenet_v2_trained_classifier.hd5', monitor='val_loss',
                     save_best_only=True)

# Fit the model using the generators and the callbacks above defined.
model.fit_generator(traingen,
                    epochs=EPOCHS,
                    shuffle=True,
                    callbacks=[rlr, mc, es],
                    validation_data=validgen)

# Trained model's score.
scores = model.evaluate(validgen, verbose=1)
print('Validation loss: ', scores[0])
print('Validation accuracy: ', scores[1])
