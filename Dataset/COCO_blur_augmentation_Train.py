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

parent_path = '/fs1/project/iip/varelal/'                          #project path
COCO_path =   parent_path + 'COCO_dataset/train2014/train2014/'     #COCO path
CSV_path =     parent_path + '/COCO_blurred_V1/train_dataset.csv'   #path to save csv file   
COCO_output = parent_path + '/COCO_blurred_V1/Train/'                 #Output of blurred images
blurr_Params_dir = parent_path + 'Blurr_Kernel_Parameters.csv'      #Blurr parameter pairs

n_cores = 2         #Number of cores to use for parallel use

#%% For Loops
def Data_Augmentation(COCO_dir):
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
    COCO_dir = sorted(glob.glob(COCO_path + '*'))[15000:16000]
    print("\tFound {} Images in dataset".format(len(COCO_dir)))
    print("\tStarting to blur")
    
    
    start = time.time()
    
    #Open CSV file
    if os.path.exists(CSV_path) == False:
        CSV_file = open(CSV_path, mode='a', newline='')
        csv_writer = csv.writer(CSV_file, delimiter=',', quotechar='"')
        csv_writer.writerow(['filename','length','angle'])
        CSV_file.close()
    
    #Split images into cores
    split_glob = np.array_split(COCO_dir,n_cores)
    
    pool = Pool(n_cores)
    pool.map(Data_Augmentation, split_glob)
    pool.close()
    
    elapsed = time.time() - start
    print("It took {} hours to run".format(elapsed/3600))