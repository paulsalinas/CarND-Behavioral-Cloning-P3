import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

root_path = './data/bend_1'
csv_path = root_path + '/driving_log.csv'
lines = []

with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    steering_value = line[3]
    filename = source_path.split('/')[-1]
    current_path = root_path + '/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(steering_value)

X_train = np.array(images)
y_train = np.array(measurements)

normalize = lambda x: x / 255 - 0.5

model = Sequential()

model.add(Lambda(normalize, input_shape=(160, 320,3)))

model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, shuffle=True)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')