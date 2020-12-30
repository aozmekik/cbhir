import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.spatial

class_names = ['Rural Area', 'Urban Area', 'Cultivated Land', 'Forest']


def read_hsi(file):
    return scipy.io.loadmat(file)['patch']

def get_hsi(dir='dataset/AnkaraHSIArchive'):
    X, Y, names, clss = [], [], [], []

    land_use_categories = pd.read_excel('dataset/AnkaraHSIArchive/Labels.xlsx',
                                        sheet_name='Land-Use Categories')
    land_cover_classes = pd.read_excel('dataset/AnkaraHSIArchive/Labels.xlsx',
                                       sheet_name='Land-Cover Classes')

    for file in os.listdir(dir):
        if file.endswith('.mat'):
            name = os.path.join(dir, file)
            i = int(file.split('_')[0]) - 1
            # FIXME. double check.
            # print(file, i, land_use_categories.iat[i, 0],
            #       land_use_categories.iat[i, 1], land_use_categories.iat[i, 1] - 1)
            # print(land_cover_classes.iloc[i])
            X.append(read_hsi(name))
            Y.append(land_use_categories.iat[i, 1] - 1)
            names.append(name)
            clss.append([int(row_name == 'x')
                         for _, row_name in land_cover_classes.iloc[i].iteritems()][1:])
    return np.array(X), np.array(Y), names, clss


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
    db = [i[0] for i in sorted(
        enumerate(list(db)), key=lambda p: np.linalg.norm(p[1]-query))]
    return db[:k]


def score(db_clss, result):
    # number of retrieved images (-1, because the first one is the query)
    R = len(result) - 1

    # category labels associated with query image
    Lq = db_clss[result[0]]

    # category labels associated with retrieved images
    LXR = [db_clss[i] for i in result[1:]]

    # set of category labels associated to archive
    LX = 4

    ac, pr, rc, hl = 0, 0, 0, 0
    for LXr in LXR:
        intersect = len(
            [1 for index, label in enumerate(LXr) if label == 1 and Lq[index] == 1])
        union = len(
            [1 for index, label in enumerate(LXr) if label == 1 or Lq[index] == 1])
        ac += intersect / union
        pr += intersect / len([1 for l in LXr if l == 1])
        rc += intersect / len([1 for l in Lq if l == 1])
        hl += len([1 for index, label in enumerate(LXr) if (label ==
                                                            1 and Lq[index] == 0) or (label == 0 and Lq[index] == 1)])
    return ac/R, pr/R, rc/R, hl/R


def inference(query, db_feature=None, db_names=None, db_clss=None, model='hsi_model', verbose=False):
    if not db_names:
        db_img, _, db_names, db_clss = get_hsi()
        model = models.load_model(model)
        features = models.Model(
            inputs=model.input, outputs=model.layers[-2].output)
        db_feature = features.predict(db_img)

    query_feature = db_feature[db_names.index(query)]
    closest = k_closest(db_feature, query_feature)
    ac, pr, rc, hl = score(db_clss, closest)
    if verbose:
        print('Retrieved images: ')
        for i in closest:
            print('\t>' + db_names[i][:-3] + 'bmp')
        print('AC (%): {:.2f}\nPR (%): {:.2f}\nRC (%): {:.2f}\nHL    : {:.2f}'.format(
            ac, pr, rc, hl))
    return ac, pr, rc, hl


def cbir(model='hsi_model'):
    db_img, _, db_names, db_clss = get_hsi()
    model = models.load_model(model)
    features = models.Model(
        inputs=model.input, outputs=model.layers[-2].output)
    db_feature = features.predict(db_img)

    AC, PR, RC, HL = 0, 0, 0, 0
    for query in db_names:
        ac, pr, rc, hl = inference(query, db_feature, db_names, db_clss, model)
        AC += ac
        PR += pr
        RC += rc
        HL += hl
    N = len(db_names)
    AC, PR, RC, HL = AC/N, PR/N, RC/N, HL/N
    print('AC (%): {:.2f}\nPR (%): {:.2f}\nRC (%): {:.2f}\nHL    : {:.2f}'.format(
        AC, PR, RC, HL))

def show_img(X_train, Y_train):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i][:, :, 10], cmap=plt.cm.binary)
        plt.xlabel(class_names[Y_train[i]])
    plt.show()

def retrieve(query):
    # TODO. test on closest
    db_img, db_label, db_name = get_hsi()
    model = models.load_model('hsi_model')
    # model.summary()
    features = models.Model(inputs=model.input, outputs=model.layers[-2].output)

    db_feature = features.predict(db_img)
    query_feature = db_feature[db_name.index(query)]
    closest = k_closest(db_feature, query_feature)
    print('Retrieved images: ')
    for i in closest:
        print(db_name[i][:-3] + 'bmp')
        
    plt.figure(figsize=(10, 10))

    for i in range(6):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(db_img[closest[i]][:, :, 10], cmap=plt.cm.binary)
        plt.xlabel(class_names[db_label[i]])
    plt.show()

# cbir()


# img = scipy.io.loadmat(
#     'dataset/AnkaraHSIArchive/001_EO1H1770322015230110KF_Radiance_1x1.mat')

# img = img['patch']
# print(img)

# img = img['patch']

# view = imshow(img, (29, 19, 10))

# plt.pause(10)
