# Circle Detection

## Overview

### Goal
The goal of this challenge was to create a image regression model for determining the center point and radius of a circle in a noisy image.

### Approach
To achieve this, given the fact that it specified that we should use CNN's in the description I used a deep Convolutional Neural Network with multiple convolutional blocks, the ReLU activation function and Batch Normalization.

## Code Walkthrough

### Environment
For this project, Anaconda (specifically MiniConda) was the environment manager of choice. If this is not installed refer to this [link](https://docs.anaconda.com/free/anaconda/install/index.html). After that is done, you can run `conda env create -f docs/<OS>-requirements.yml` where OS is either **linux** or **mac**. Note there is no requirements for windows as I do not have access to a windows computer.

### Dataset
To start using this GitHub repository and train your own, you first need to create a dataset. To generate this dataset (due to GitHub's file limitations I could not push this), you can simply refer under `circle/notebooks/exploration.ipynb` and execute the **Import** and **Dataset** code block sections sequentially and this will create a diverse, balanced and generalizable dataset of noisy cicles, `circle/data/train/images.npy`, and it's respective labels, `circle/data/train/labels.npy`. This will also generate a test dataset for mixed testing under `circle/data/test`.  Note there is an additional section in the notebook related to data exploration called **Exploration** to see visually how changes in noise level influence the quality of picture.

### Model
To see the model definition you can refer to `circle/regression.py`. It uses PyTorch for it's implementation.

### Training
In order to train the model you can run the command `python3 circle/train.py` from the main directory. Note this uses argparse to control the various hyperparameters so if you wish to see what you can change run `python3 circle/train.py --help`. All checkpoints of the model will be placed under `circle/checkpoints`. In addition as training occurs a log file will be automatically updated called `new_out.txt` so you can refer to your models performance without needing to refer to the terminal.

### Testing & Evaluation
In order to get an accurate idea of how well your model did in training you can run the command `python3 circle/test.py` and this will print out in the terminal how the model you chose did with respect to the testing set as well as on a noise level basis.