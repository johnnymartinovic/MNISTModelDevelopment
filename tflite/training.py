from __future__ import print_function

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from matplotlib import pyplot as plt

(x_train, y_train), (x_val, y_val) = mnist.load_data()

# Inspect x data
print('x_train shape: ', x_train.shape)
# Displays (60000, 28, 28)
print(x_train.shape[0], 'training samples')
# Displays 60000 train samples
print('x_val shape: ', x_val.shape)
# Displays (10000, 28, 28)
print(x_val.shape[0], 'validation samples')
# Displays 10000 validation samples

# print('First x sample\n', x_train[0])
# Displays an array of 28 arrays, each containing 28 gray-scale values between 0 and 255
# Plot first x sample
plt.imshow(x_train[0])
plt.show()

# Inspect y data
print('y_train shape: ', y_train.shape)
# Displays (60000,)
print('First 10 y_train elements:', y_train[:10])
# Displays [5 0 4 1 9 2 1 3 1 4]

img_rows, img_cols = x_train.shape[1], x_train.shape[2]
num_classes = 10

# Set input_shape for channels_first or channels_last
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
# x_train shape: (60000, 28, 28, 1)
print('x_val shape:', x_val.shape)
# x_val shape: (10000, 28, 28, 1)
print('input_shape:', input_shape)
# input_shape: (28, 28, 1)

# print('First x sample, normalized\n', x_train[0])
# An array of 28 arrays, each containing 28 arrays, each with one value between 0 and 1

model_m = Sequential()
model_m.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
model_m.add(MaxPooling2D(pool_size=(2, 2)))
model_m.add(Dropout(0.5))
model_m.add(Conv2D(64, (3, 3), activation='relu'))
model_m.add(MaxPooling2D(pool_size=(2, 2)))
model_m.add(Dropout(0.2))
model_m.add(Conv2D(128, (1, 1), activation='relu'))
model_m.add(MaxPooling2D(pool_size=(2, 2)))
model_m.add(Dropout(0.2))
model_m.add(Flatten())
model_m.add(Dense(128, activation='relu'))
model_m.add(Dense(num_classes, activation='softmax'))
# Inspect model's layers, output shapes, number of trainable parameters
print(model_m.summary())

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
batch_size = 32
epochs = 10

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
model_m.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs)

# Exporting to TFLite model
import tensorflow as tf

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model_m)
tflite_model = converter.convert()

# Save the model.
with open('./model.tflite', 'wb') as f:
    f.write(tflite_model)
