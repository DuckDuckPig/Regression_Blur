# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:36:11 2020
V3 - Fixes blurr kernel labeling using a pandas dataframe
@author: varel
"""
#%% Imports
import numpy as np
import os
import glob
from Motion_blur import Motion_blur
import progressbar
import imageio as io
import csv
from multiprocessing import Pool
import time
import pandas as pd

parent_path = 'D:\\\\' #Parent directory
COCO_path =   parent_path + 'COCO_dataset/val2014/val2014/'     #COCO path for val images
CSV_paths =     [parent_path + '/COCO_blurred_V4/test_dataset.csv', parent_path + '/COCO_blurred_V4/val_dataset.csv']   #paths for csv train, validation   
COCO_outputs = [parent_path + '/COCO_blurred_V4/Test/', parent_path + '/COCO_blurred_V4/Validate/']                 # paths to save images for train and validation
blurr_Params_dir = parent_path + 'Blurr_Kernel_Parameters.csv'      #'D:\\Blurr_Kernel_Parameters.csv'



#%% For Loops
def Data_Augmentation(COCO_dir, CSV_path, COCO_output):
    # directories
    print("\tOpening Blurr Parameters")
    blurr_Params = pd.read_csv(blurr_Params_dir)

    CSV_file = open(CSV_path, mode='a', newline='')
    csv_writer = csv.writer(CSV_file, delimiter=',', quotechar='"')
    
    #Create COCO_dir as an iteration
    COCO_dir = iter(COCO_dir)
    
    #Progressbar
    # bar = progressbar.ProgressBar(maxval=len(COCO_dir), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # bar.start()
    # bar.update(0)

    for r in range(1,101):
        count = 0
        print("R is: ",r)
        new_table = blurr_Params[blurr_Params['Length'] == r]
        angles = new_table['Angle'].to_numpy()        
        
        while count < 175:
            
            COCO = next(COCO_dir)
            Image = np.asarray(io.imread(COCO))
            basename = os.path.basename(COCO).split('_')[-1].split('.')[0]
            
            for angle in angles:
                count += 1
                #Blurr image and get correct angle and length after K change    
                I_blurr, n_r, n_angle = Motion_blur(Image, r, angle)
        
                #Create new filename and path as well as saving image
                filename = basename + '_' + str(r) + '_' + str(angle) + '.png'
                full_path = COCO_output + filename
                io.imsave(full_path, I_blurr.astype('uint8'))
                csv_writer.writerow([str(filename),str(r),str(angle)])
        
            
        # bar.update(t+1)
    CSV_file.close()
    # Pandas to csv
    #Train_dataframe.to_csv(COCO_csv, index=False, mode='a', header=False) 
if __name__ == '__main__':    
    print("\tGetting images")

    #COCO_txt = 'C:\\Users\\varel\\Documents\\Research\\train_directory.txt'
    COCO_dir = sorted(glob.glob(COCO_path + '*'))
    print("\tFound {} Images in dataset".format(len(COCO_dir)))
    print("\tStarting to blur")
    n_cores = 2      # Number of cores to parallel training and validation creation. This shouldnt change 
    
    start = time.time()
    
    #Open CSV file
    for CSV_path in CSV_paths:
        if os.path.exists(CSV_path) == False:
            CSV_file = open(CSV_path, mode='a', newline='')
            csv_writer = csv.writer(CSV_file, delimiter=',', quotechar='"')
            csv_writer.writerow(['filename','length','angle'])
            CSV_file.close()
    
    #Split images into cores
    split_glob = np.array_split(COCO_dir,n_cores)
    CSV_path = np.array_split(CSV_paths, n_cores)
    COCO_output = np.array_split(COCO_outputs, n_cores)
    
    data = [(split_glob[0], CSV_path[0], COCO_output[0]),(split_glob[1],CSV_path[1], COCO_output[1])]
    
    pool = Pool(n_cores)
    pool.starmap(Data_Augmentation, zip(split_glob, CSV_path, COCO_output))
    pool.close()
    
    elapsed = time.time() - start
    print("It took {} hours to run".format(elapsed/3600))