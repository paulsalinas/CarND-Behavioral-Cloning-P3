from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from itertools import chain

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils import generator, get_samples, get_sample_image

root_path = './data/bend_1'

train_samples, validation_samples = train_test_split(get_samples(root_path), test_size=0.2)

# plt.imshow(get_sample_image(root_path, train_samples))
# plt.show()

# plt.imshow(np.fliplr(get_sample_image(root_path, train_samples)))
# plt.show()

train_generator = generator(
    root_path, 
    train_samples) 

validation_generator = generator(
    root_path, 
    validation_samples) 

flipped_train_generator = generator(
    root_path, 
    train_samples, 
    aug_image_fn=lambda x: np.fliplr(x),
    aug_measurement_fn=lambda x: x * -1)

flipped_validation_generator = generator(
    root_path, 
    validation_samples, 
    aug_image_fn=lambda x: np.fliplr(x),
    aug_measurement_fn=lambda x: x * -1)

train_generator = chain(train_generator, flipped_train_generator)
validation_generator = chain(validation_generator, flipped_validation_generator)

# image normalization function
normalize = lambda x: x / 255 - 0.5

# begin network
model = Sequential()

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(normalize))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2),  activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2),  activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2),  activation='relu'))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

num_train_samples = len(train_samples) * 3 * 2
num_valid_samples = len(validation_samples) * 3 * 2

history_object = model.fit_generator(
    train_generator, 
    validation_data=validation_generator, 
    samples_per_epoch=num_train_samples,
    nb_val_samples=num_valid_samples,
    nb_epoch=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model.h5')
