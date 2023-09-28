# Regression_Blur
This code implements a model to predict linear motion blur parameters using a regresion Convolutional Neural Network. 

## Requirements
The `Requirements.txt` file is based of a conda environment. To create a conda environment based on Requirements.txt 

> conda create --name <env> --file requiremnts.txt

## Dataset
This folder has the scripts to create the blur datasets. The dataset is based of [COCO 2014](https://cocodataset.org/#download)

`COCO_blur_augmentation_TestVal` and `COCO_blur_augmentation_Train` create the Test/Val and Train datasets to train the model. The code randomises COCO images which means it wont create the exact blurred images we used. The code is parallelized to use multiple cores. The number of images you get in train will depend on the number of cores you use. Each core yields approximately 21K images.

`COCO_blur_using_csv.py` reads in one of the csv files to create the exact blur images. This is a slower method of creating images which will not be optimal for the training models, but can be optimal to test with some of the images.

## Regression Blur

