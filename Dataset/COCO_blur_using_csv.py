# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:51:58 2023

@author: varel
"""

#%% Imports
import numpy as np
import os
import glob
from Motion_blur import Motion_blur
import imageio as io
import csv
from multiprocessing import Pool
import time
import pandas as pd

parent_path = '/fs1/project/iip/varelal/'                          #project path
COCO_path =   'D:/COCO_dataset/val2014/val2014/'     #COCO path
COCO_output = parent_path + '/COCO_blurred_V1/L5_dataset/'                 #Output of blurred images
CSV_file = 'A3_Subset_Dataset.csv'
blurr_Params_dir = parent_path + 'Blurr_Kernel_Parameters.csv'      #Blurr parameter pairs


#%%
df = pd.read_csv(CSV_file)
filenames = np.asarray(df['filename'])

for filename in filenames[0:1]:
    img_name, length, angle = filename.split('.')[0].split('_')
    
    img_name = 'COCO_val2014_' + img_name + '.jpg'
    
    #Read in image
    I = io.imread(COCO_path + img_name)
    
    #Blur image
    I_blur, nr, nangle = Motion_blur(I, int(length), int(angle))
    
    io.imsave(COCO_output + filename, I_blur)
    
