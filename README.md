# Regression_Blur
This code implements a model to predict linear motion blur parameters using a regression Convolutional Neural Network. 

## Requirements
The `Requirements.txt` file is based off a conda environment. To create a conda environment based on Requirements.txt 

> conda create --name <env> --file requiremnts.txt

## Dataset
This folder has the scripts to create the blur datasets. The dataset is based off [COCO 2014](https://cocodataset.org/#download)

`COCO_blur_augmentation_TestVal` and `COCO_blur_augmentation_Train` create the Test/Val and Train datasets to train the model. The code randomises COCO images which means it wont create the exact blurred images we used. The code is parallelized to use multiple cores. The number of images you get in train will depend on the number of cores you use. Each core yields approximately 21K images. The code will also export a csv file with the labeled data for filenames, angle, and length blur parameters. 

`COCO_blur_using_csv.py` reads in one of the csv files to create the exact blur images. This is a slower method of creating images which will not be optimal for the training datasets, but can be optimal to test with the subset datasets.

# Regression Blur
`VGG16_Regressor.py` Trains the regressor model. 
  ``parent_dir`` variable needs to point to the directory where all your blur datasets are located. 
  ``train_dir``, ``val_dir``, ``test_dir`` are the variables that concatenated the ``parent_dir`` with the specified folder for each dataset.
    ``train_labels``, ``val_labels``, test_labels`` are the csv file names for each dataset that will be used by the data generator to create the batches with its labeled data.

`Results.py` Will run the results of your trained model, and calculate the $R^2$ score for each blur parameter. May test multiple saved weights at the same time. Need to change ``weights_filenames`` for the weights you enter.

# Deconvolution
The deconvolution method was implemented from Zoran using this [EPLL code](https://github.com/friedmanroy/torchEPLL)
