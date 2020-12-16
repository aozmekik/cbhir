import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
import matplotlib.pyplot as plt

# TODO. get a keras model (CNN?) and write it.
# TODO. give dataset to the model. get some results.

# TODO. get a feature vector from keras model
# TODO. write a cbir system.


def get_hsi(ratio=.1, dir='dataset/AnkaraHSIArchive'):
    X, Y = [], []

    dfs = pd.read_excel('dataset/AnkaraHSIArchive/Labels.xlsx',
                        sheet_name='Land-Use Categories')

    for file in os.listdir(dir):
        if file.endswith('.mat'):
            X.append(scipy.io.loadmat(os.path.join(dir, file))['patch'])
            i = int(file.split('_')[0]) - 1
            # print(file, i, dfs.iat[i, 0], dfs.iat[i, 1]) # FIXME. double check.
            Y.append(dfs.iat[i, 1] - 1)
    return train_test_split(np.array(X), np.array(Y), test_size=ratio, random_state=42)


def show_img(X_train, Y_train):
    plt.figure(figsize=(10, 10))
    class_names = ['Rural Area', 'Urban Area', 'Cultivated Land', 'Forest']

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i][:, :, 10], cmap=plt.cm.binary)
        plt.xlabel(class_names[Y_train[i]])
    plt.show()


def create_model():
    num_classes = 4

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(63, 63, 119)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))
    model.summary()


def train_model(model, X_train, Y_train, X_test, Y_test):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=10,
                        validation_data=(X_test, Y_test))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


# print(dfs)
X_train, X_test, Y_train, Y_test = get_hsi()
print(Y_train)
# show_img(X_train, Y_train)
model = create_model()
train_model(model, X_train, Y_train, X_test, Y_test)
# print(data)


# img = scipy.io.loadmat(
#     'dataset/AnkaraHSIArchive/001_EO1H1770322015230110KF_Radiance_1x1.mat')

# img = img['patch']
# print(img)

# img = img['patch']

# view = imshow(img, (29, 19, 10))

# plt.pause(10)
