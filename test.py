import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.models import Sequential
from keras import datasets
import keras
import os
# if like me you do not have a lot of memory in your GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# then these two lines force keras to use your CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:, :-1]


def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
    return np.asarray(data_t, dtype=np.float32)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(train_images.shape, train_labels.shape)

# with h5py.File("./dataset/MNIST/full_dataset_vectors.h5", "r") as hf:

#     # Split the data into training/test features/targets
#     X_train = hf["X_train"][:]
#     targets_train = hf["y_train"][:]
#     X_test = hf["X_test"][:]
#     targets_test = hf["y_test"][:]
#     print(X_train.shape, targets_test.shape, X_test.shape, targets_test.shape)

#     # Determine sample shape
#     sample_shape = (16, 16, 16, 3)

#     # Reshape data into 3D format
#     X_train = rgb_data_transform(X_train)
#     X_test = rgb_data_transform(X_test)

#     print(X_train.shape, X_test.shape, targets_test.shape)


#     # Convert target vectors to categorical targets
#     targets_train = to_categorical(targets_train).astype(np.integer)
#     targets_test = to_categorical(targets_test).astype(np.integer)
