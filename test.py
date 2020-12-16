import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.spatial

# TODO. calculate all the prediction values.


def read_hsi(file):
    return scipy.io.loadmat(file)['patch']


def get_hsi(dir='dataset/AnkaraHSIArchive'):
    X, Y, names = [], [], []

    dfs = pd.read_excel('dataset/AnkaraHSIArchive/Labels.xlsx',
                        sheet_name='Land-Use Categories')

    for file in os.listdir(dir):
        if file.endswith('.mat'):
            name = os.path.join(dir, file)
            X.append(read_hsi(name))
            i = int(file.split('_')[0]) - 1
            # print(file, i, dfs.iat[i, 0], dfs.iat[i, 1]) # FIXME. double check.
            Y.append(dfs.iat[i, 1] - 1)
            names.append(name)
    return np.array(X), np.array(Y), names


def split_hsi(X, Y, ratio=.1):
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


def k_closest(db, query, k=6):
    db = [i[0] for i in sorted(enumerate(list(db)), key = lambda p: np.linalg.norm(p[1]-query))]
    return db[:k]

def cbir(query):
    # TODO. test on closest
    db_img, _, db_tag = get_hsi()
    model = models.load_model('hsi_model')
    # model.summary()
    features = models.Model(
        inputs=model.input, outputs=model.layers[-2].output)

    db_feature = features.predict(db_img)
    query_feature = db_feature[db_tag.index(query)]
    closest = k_closest(db_feature, query_feature)
    print('Retrieved images: ')
    for i in closest:
        print(db_tag[i][:-3] + 'bmp')



# print(dfs)
# X_train, X_test, Y_train, Y_test = get_hsi()
# print(Y_train)
# # show_img(X_train, Y_train)
# model = create_model()
# train_model(model, X_train, Y_train, X_test, Y_test)
# # print(data)
# cbir('dataset/AnkaraHSIArchive/099_EO1H1770322015230110KF_Radiance_25x3.mat')


# img = scipy.io.loadmat(
#     'dataset/AnkaraHSIArchive/001_EO1H1770322015230110KF_Radiance_1x1.mat')

# img = img['patch']
# print(img)

# img = img['patch']

# view = imshow(img, (29, 19, 10))

# plt.pause(10)
