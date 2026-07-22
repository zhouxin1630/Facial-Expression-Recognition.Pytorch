# Facial-Expression-Recognition.Pytorch

A CNN-based PyTorch implementation for facial expression recognition on FER2013 and CK+ datasets.

This repository has been updated to be compatible with Python 3.10 and PyTorch 2.x.

## Demos
![Image text](https://raw.githubusercontent.com/WuJie1010/Facial-Expression-Recognition.Pytorch/master/demo/1.png)
![Image text](https://raw.githubusercontent.com/WuJie1010/Facial-Expression-Recognition.Pytorch/master/demo/2.png)

## Environment requirements
The code in this repository has been verified under the following environment:

- Python 3.10.x
- PyTorch 2.x (tested with 2.13.0+cu126)
- torchvision 0.28.x
- h5py
- numpy
- Pillow
- scikit-learn
- matplotlib

## Installation
If you use conda, a typical setup is:

```bash
conda create -n emotion_env python=3.10
conda activate emotion_env
pip install "numpy<2" h5py pillow scikit-learn matplotlib
pip install torch torchvision
```

If you already use the environment in this workspace, the dependencies are already installed and ready to use.

## Project structure
- `mainpro_FER.py`: training and evaluation for FER2013
- `mainpro_CK+.py`: training and evaluation for CK+
- `plot_fer2013_confusion_matrix.py`: confusion matrix for FER2013
- `plot_CK+_confusion_matrix.py`: confusion matrix for CK+
- `visualize.py`: inference on a single image with a pretrained checkpoint
- `upgrade_notes_python3_torch2.md`: change log for the Python 3 / PyTorch 2 migration

## FER2013 dataset
Dataset source:
- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Image properties:
- 48 x 48 pixels
- labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

The training set contains 28,709 examples, the public test set contains 3,589 examples, and the private test set contains another 3,589 examples.

### Prepare FER2013 data
1. Download `fer2013.csv`.
2. Put it into the `data` folder.
3. Run:

```bash
python preprocess_fer2013.py
```

### Train and evaluate FER2013
```bash
python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01
```

### Plot FER2013 confusion matrix
```bash
python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest
```

## CK+ dataset
The CK+ dataset is an extension of the CK dataset. It contains 327 labeled facial videos. The project extracts the last three frames from each sequence and uses 10-fold cross validation.

### Prepare CK+ data
The dataset file `CK_data.h5` is expected under the `data` directory.

### Train and evaluate one CK+ fold
```bash
python mainpro_CK+.py --model VGG19 --bs 128 --lr 0.01 --fold 1
```

### Train and evaluate all CK+ folds
```bash
python k_fold_train.py
```

### Plot CK+ confusion matrix
```bash
python plot_CK+_confusion_matrix.py --model VGG19
```

## Visualize a test image with a pretrained model
1. Put a pretrained checkpoint into the `FER2013_VGG19` folder.
2. Put your test image as `images/1.jpg`.
3. Run:

```bash
python visualize.py
```

## Notes for the new PyTorch version
The scripts in this repository have been updated to work with PyTorch 2.x. The main changes include:

- replacing `Variable` and `volatile=True`
- using `device` instead of `.cuda()` directly
- using `torch.no_grad()` for inference
- saving/loading checkpoints as `.pth` files

## Expected accuracy (reference)
The original project reported the following approximate results:

- FER2013 with VGG19:
  - PublicTest accuracy: about 71.5%
  - PrivateTest accuracy: about 73.1%

- CK+ with VGG19:
  - Test accuracy: about 94.6%

