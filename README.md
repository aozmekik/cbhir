# Content-Based Hyperspectral Image Retrieval

Content-based image retrieval on the AnkaraHSI hyperspectral archive. A 2D or 3D CNN is trained to classify the 216 image patches by land use, and the output of its last hidden layer is used as the image descriptor: a query patch returns the five patches closest to it in that feature space.

Undergraduate graduation project (BİL 495) in the Department of Computer Engineering at Gebze Technical University, supervised by Prof. Dr. Erchan Aptoula, January 2021. The report, [rapor.pdf](rapor.pdf), is in Turkish with an English summary.

## How it works

- `get_hsi()` in `utils.py` loads every `.mat` patch in `dataset/AnkaraHSIArchive` (63 × 63 pixels, 119 bands), its land-use category (Rural Area, Urban Area, Cultivated Land, Forest) and its multi-label land-cover vector from `Labels.xlsx`.
- The network is trained as a 4-class land-use classifier (sparse categorical cross-entropy, Adam) on an 80/10/10 train/validation/test split.
  - 2D CNN (`cnn2d.ipynb`): the 119 bands are input channels. Three 3×3 `Conv2D` layers (32, 64, 64 filters) with max pooling, then `Dense(64)` and `Dense(4)`.
  - 3D CNN (`cnn3d.ipynb`): the patch is a 63×63×119×1 volume. Three 3×3×3 `Conv3D` layers (32, 64, 64 filters) with max pooling, then `Dense(128)`, `Dense(64)` and `Dense(4)`.
- Retrieval: the output of `model.layers[-2]` (the last hidden `Dense(64)`) is the feature vector. `k_closest()` sorts the archive by distance to the query under the L0, L1, L2 or L∞ norm and keeps the query plus the 5 nearest patches.
- Scoring: `score()` compares each retrieved patch with the query on the land-cover labels and averages accuracy (intersection over union of the label sets), precision, recall and Hamming loss. `cbir()` uses every patch in the archive as a query once and averages the results.
- `multiple_test()` in both notebooks retrains on 5 different random splits and averages test accuracy and retrieval scores for each norm. In the saved outputs, the 3D CNN features score better than the 2D CNN features on all four metrics.

The report also covers the literature (endmember-based methods), a HybridSN model, and a comparison with the bag-of-endmembers results of Ömrüuzun et al. (2016).

## Running

The notebooks were run on Google Colab: the first cell mounts Google Drive and changes into the project folder. Elsewhere, skip that cell and start Jupyter in the repository root, because `utils.py` opens `dataset/AnkaraHSIArchive` by relative path.

Imported packages: `tensorflow`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`; `other.py` also needs `PySimpleGUI` and `Pillow`. The notebook metadata records Python 3.8.6, and the committed models were written by TensorFlow 2.3.0. `utils.py` calls `Series.iteritems()`, which pandas 2.0 removed, so an older pandas is needed.

1. `cnn2d.ipynb`, `cnn3d.ipynb`: train a model, evaluate it on the test split and save it as `cnn2d_model` or `cnn3d_model`.
2. `cbhir.ipynb`: load a saved model and print the averaged scores, e.g. `cbir('cnn3d_model', norm='l2')`.
3. `other.py`: a small PySimpleGUI window ("Hyperspectral CBIR Demo") that lists the patches, runs a query with the L2 norm and shows the 5 retrieved patches with their scores. It loads `cnn3d_model` from the working directory.

## Layout

- `utils.py`: data loading, retrieval and scoring functions
- `cnn2d.ipynb`, `cnn3d.ipynb`, `cbhir.ipynb`: experiments, with outputs
- `other.py`: GUI demo
- `cnn_model/`, `hsi_model/`: two saved 2D CNN models (TensorFlow SavedModel)
- `dataset/AnkaraHSIArchive/`: 216 `.mat` patches with `.bmp` previews, `Labels.xlsx` and the archive's own `Readme.txt`
- `scores.xlsx`: score sheet for the experiments
- `rapor.pdf`: project report (Turkish)
- `presentations/`: the four project presentations (Turkish), PDF and PPTX
- `test.py`: an early Keras scratch file that loads CIFAR-10; not used by the notebooks

## Dataset

The AnkaraHSIArchive is a small benchmark archive of 216 tiles cut from an EO-1 Hyperion image of the Ankara area (August 2015, 30 m ground resolution, 119 bands left after removing low-SNR and defective channels). Each tile has several land-cover class labels and one land-use category. The copy here includes the archive's `Readme.txt`, which describes it as published for research purposes and asks that publications using it cite:

> F. Ömrüuzun, B. Demir, L. Bruzzone and Y. Y. Çetin, "Content based hyperspectral image retrieval using bag of endmembers image descriptors," 2016 8th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), Los Angeles, CA, 2016, pp. 1-4.

## Notes

- `cnn2d_model` and `cnn3d_model` are created by the notebooks and are not committed. `cbhir.ipynb` defaults to `hsi_model`, and `other.py` and `utils.cbir()` default to `cnn3d_model`.
- The HybridSN experiments described in the report are not in this repository.
