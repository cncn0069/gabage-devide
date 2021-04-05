import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'C:\\Users\\HP\\PycharmProjects\\pythonProject\\train'
val_dir = 'C:\\Users\\HP\\PycharmProjects\\pythonProject\\val'

batch_size = 30
img_height = 150
img_width = 150

train_da = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=70,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    zca_epsilon=True,


)

train_ds = train_da.flow_from_directory(
    train_dir,
    #subset='training',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=batch_size
)

val_da = ImageDataGenerator(rescale=1. / 255)

val_ds = val_da.flow_from_directory(
    val_dir,
    #subset='validation',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=batch_size
)

from tensorflow.keras import layers
from tensorflow.keras.applications import *

vgg16.trainable = False
model = tf.keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='gelu',input_shape=(img_height, img_width, 3), kernel_initializer='glorot_uniform'),
    layers.Conv2D(64, (3, 3), activation='gelu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3, 3), activation='gelu'),
    layers.Conv2D(128, (3, 3), activation='gelu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='gelu'),
    layers.Conv2D(256, (3, 3), activation='gelu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation='gelu'),
    layers.Conv2D(512, (3, 3), activation='gelu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(512, activation='gelu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='gelu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='gelu'),
    layers.Dense(6, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

import datetime, os

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)
EarlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.1, patience=30, verbose=0, mode=min, baseline=None, restore_best_weights=True
)
history = model.fit(
    train_ds,
    batch_size=batch_size,
    validation_data=val_ds,
    epochs=10000,
    callbacks=[tensorboard_callback, cp_callback, EarlyStop]
)
model.save('garbage.h5')

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()
