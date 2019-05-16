# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:19:01 2019

@author: SS066237
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 11:28:30 2018

@author: SS066237
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, Flatten, Dense

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('your directory',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('your directory',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

# get the class lebels for the training data, in the original order

from keras.utils.np_utils import to_categorical
train_labels = training_set.classes
num_classes = len(training_set.class_indices)
    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = test_set.classes
num_classes1 = len(test_set.class_indices)
    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
test_labels = to_categorical(test_labels, num_classes=num_classes1)

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(num_classes, activation = 'softmax'))
# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])




from keras.callbacks import EarlyStopping
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, mode='auto', verbose=1)

# do the actual fitting
autoencoder_train = classifier.fit_generator(
        training_set,
        steps_per_epoch = 1487,
        validation_data=test_set,
        epochs=10,
        validation_steps = 520,
        shuffle=False,
        callbacks=[stopper]
        )


