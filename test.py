from spectral import *
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

# TODO list
# TODO. write a dataset loader & normalizer
# TODO. get a keras model (CNN?) and write it.
# TODO. give dataset to the model. get some results.

# TODO. get a feature vector from keras model
# TODO. write a cbir system.


key = 'indian_pines_corrected'
key_gt = 'indian_pines_gt'


img = scipy.io.loadmat(
    'dataset/AnkaraHSIArchive/001_EO1H1770322015230110KF_Radiance_1x1.mat')


img = img['patch']
print(img.shape)

# img = img['patch']

# view = imshow(img, (29, 19, 10))

# plt.pause(10)
