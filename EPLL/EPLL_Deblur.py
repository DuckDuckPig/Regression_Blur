# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:41:59 2023

@author: varel
"""

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import conv2d
from matplotlib import pyplot as plt
from skimage.draw import line
from tqdm import tqdm
import os
import glob
import csv
#from multiprocessing import Pool

import sys
sys.path.append('torchEPLL') #From Github
from models import GMM, GMMDenoiser
from EPLL import decorrupt
from time import time
torch.set_grad_enabled(False)
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim

from torch.multiprocessing import set_start_method, Pool, Process

import warnings
warnings.filterwarnings("ignore")


#%% Setup inputs
sharp_img_dir = '/home/varelal/Documents/COCO_dataset/val2014/COCO_val2014_'
blur_img_dir = '/home/varelal/Documents/COCO_blurred_V1/Test/'
filedata = '/home/varelal/Documents/Uniform_blurr/VGG16_model_fits/Test_Subset_Predictions_A3.csv'
CSV_path = 'EPLL_VGG16_Results_A3.csv'
output_true = '/home/varelal/Documents/EPLL/A3_true/'
output_pred = '/home/varelal/Documents/EPLL/A3_pred/'
output_sharp = '/home/varelal/Documents/EPLL/A3_sharp/'


#%% Functions

def Motion_blur_kernel(r, angle):
    #Get the size of the Kernel
    if angle != 0 and abs(angle) != 90:
        width = int(np.ceil(np.abs(r * np.sin(angle*np.pi/180))))
        height = int(np.ceil(np.abs(r * np.cos(angle*np.pi/180))))

        if width == 0:
            width = 1
        if height == 0:
            height = 1
            
        # Define a zero sized Kernel
        K = np.zeros((width,height))
        x_max, y_max = K.shape
        #Define direction of line and special cases
        if angle < 0:
            x_start, y_start, x_stop, y_stop = [0,0,x_max-1,y_max-1]
        else:
            x_start, y_start, x_stop, y_stop = [x_max-1, 0, 0, y_max-1]
        #Draw Line
        rr, cc = line(x_start, y_start, x_stop, y_stop)
        try: K[rr,cc] = 1
        except:
            #plt.imshow(K)
            print(width,height)
            print("Max",x_max,y_max)
            print("Coordinates are ({},{}), ({},{})".format(x_start, y_start, x_stop, y_stop))
        K = K/K.sum()
        n_angle = np.arctan(width/height) * 180 /np.pi
        n_r = np.sqrt(height**2 + width**2)

        if angle < 0:
            n_angle *= -1

    #Special case angle 0    
    if angle == 0:
        K = np.ones((1,r))
        K = K/K.sum()
        n_angle = angle
        n_r = r

    #Special case angle 90 or -90
    if abs(angle) == 90:
        K = np.ones((r,1))
        K = K/K.sum()
        n_angle = angle
        n_r = r
    
    return(K)

def Motion_blur(I,r,angle):
    if len(I.shape) < 3:
        I = np.stack((I,I,I), axis=2)
        
    
    K = Motion_blur_kernel(r,angle)
    I_r = convolve2d(I[:,:,0],K, mode='same')
    I_g = convolve2d(I[:,:,1],K,mode='same')
    I_b = convolve2d(I[:,:,2],K,mode='same')
    I_blurr = np.stack((I_r,I_g,I_b), axis=2)
    return(I_blurr.astype('uint8'))

def Square_Kernel(K):
    row, col = K.shape
    if row == col:
        return K
    else:
        if row > col:
            diff = row - col

            before = diff // 2
            after = diff // 2 + np.round(diff % 2)

            return np.pad(K, ((0,0), (before,after)), 'constant', constant_values=0)

        else:
            diff = col - row

            before = diff // 2
            after = diff // 2 + np.round(diff % 2)

            return np.pad(K, ((before,after), (0,0)), 'constant', constant_values=0)
        

def EPLL(K, I, dev):
    
    gmm = GMM.load('torchEPLL/trained/GMM100.mdl')
    denoiser = GMMDenoiser(gmm.to(dev))
    
    blur_scale = 2  # bigger for blurrier images
    kernel = torch.from_numpy(K).float().to(dev)[None, None]
    H = lambda x: conv2d(x.permute(-1, 0, 1)[:, None], kernel, padding=kernel.shape[-1]//2)[:, 0].permute(1, 2, 0)

    corr = torch.from_numpy(I).to(dev).float()
    
    alpha = 1/50
    beta = lambda i: min(2**i / alpha, 3000)
    
    its = 6  # number of iterations to run the algorithm
    n_grids = 64  # the smaller this is, the faster and less accurate the algorithm will be; for original EPLL, use 64
    n = 1/255
    t = time()
    MAP = decorrupt(corr, n, H, denoiser, p_sz=8, its=its, beta_sched=beta, n_grids=n_grids, verbose=False)
    I_deblurr = np.clip(MAP.cpu().numpy(), 0, 1)
    
    
    return I_deblurr

def Deblur(I,K, dev):
    row, col = K.shape
    
    if row != col:
        raise ValueError("row and col do not match")
    
    #Checks if its an even or odd kernel. If even it will pad each corner take the EPLL with each and output the average of the image
    
    if row % 2 == 0: #Check if its even
        K1 = np.pad(K, ((1,0), (1,0)), 'constant', constant_values=0)
        K2 = np.pad(K, ((1,0), (0,1)), 'constant', constant_values=0)
        K3 = np.pad(K, ((0,1), (1,0)), 'constant', constant_values=0)
        K4 = np.pad(K, ((0,1), (0,1)), 'constant', constant_values=0)
        
        Id1 = EPLL(K1, I, dev)
        Id2 = EPLL(K2, I, dev)
        Id3 = EPLL(K3, I, dev)
        Id4 = EPLL(K4, I, dev)
        
        ID_avg = (Id1 + Id2 + Id3 + Id4) / 4
        return ID_avg
        
    else:
        Id = EPLL(K, I, dev)
        return Id

def df_chunking(df, chunksize):
    """Splits df into chunks, drops data of original df inplace"""
    count = 0 # Counter for chunks
    while len(df):
        count += 1
        print('Preparing chunk {}'.format(count))
        # Return df chunk
        yield df.iloc[:chunksize].copy()
        # Delete data in place because it is no longer needed
        df.drop(df.index[:chunksize], inplace=True)    
    
def COCO_Results(data):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    CSV_file = open(CSV_path, mode='a', newline='')
    csv_writer = csv.writer(CSV_file, delimiter=',', quotechar='"')
    

    
    for index, row in tqdm(data.iterrows()):
        filename = row['filename']
        I_blur = plt.imread(blur_img_dir + filename)

        # Read in sharp image
        sharp_file = filename.split('_')[0] + '.jpg'
        I_sharp = plt.imread(sharp_img_dir + sharp_file)/255
        #print("Unique Sharp\n", np.unique(I_sharp))

        # grab variables
        length_true =  row['length']
        angle_true = row['angle']
        
        length_pred = row['Pred_length']
        angle_pred = row['Pred_angle']
        
        # Create kernel
        K_true = Square_Kernel(Motion_blur_kernel(length_true, angle_true))
        K_pred = Square_Kernel(Motion_blur_kernel(length_pred, angle_pred))
        
        # Deconvolve blurry image
        I_decon_true = Deblur(I_blur, K_true, dev)
        I_decon_pred = Deblur(I_blur, K_pred, dev)
        
        #normalize the deconvolved image
        #I_decon_true = (I_decon_true - np.min(I_decon_true)) / (np.max(I_decon_true) - np.min(I_decon_true))
        #I_decon_pred = (I_decon_pred - np.min(I_decon_pred)) / (np.max(I_decon_pred) - np.min(I_decon_pred))

        #Save image in uint8
        plt.imsave(output_true+filename, (255*I_decon_true).astype(np.uint8))
        plt.imsave(output_pred+filename, (255*I_decon_pred).astype(np.uint8))
        plt.imsave(output_sharp+filename.split('_')[0]+'.png',(255*I_sharp).astype(np.uint8))


        #print(255*np.unique(I_decon_true))
        #print(255*np.unique(I_decon_pred))
    
        # calculate SSIM and SSD
        try:
        
            SSD_true = np.sum((I_sharp - I_decon_true) ** 2)
            SSIM_true = ssim(I_sharp, I_decon_true, channel_axis=2)
    
            SSD_pred = np.sum((I_sharp - I_decon_pred) ** 2)
            SSIM_pred = ssim(I_sharp, I_decon_pred, channel_axis=2)

            #print(SSD_true, SSD_pred)
        
            csv_writer.writerow([str(filename),str(length_true),str(angle_true),str(length_pred),str(angle_pred),str(SSD_true),str(SSIM_true),str(SSD_pred),str(SSIM_pred)])
        
        #print out or save into file
        except:
            print("Error in SSIM - filename:", filename)
            

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    

    if os.path.exists(CSV_path) == False:
        CSV_file = open(CSV_path, mode='a', newline='')
        csv_writer = csv.writer(CSV_file, delimiter=',', quotechar='"')
        csv_writer.writerow(['filename','length','angle','Pred_length','Pred_angle','True_SSD','True_SSIM','Pred_SSD','Pred_SSIM'])
        CSV_file.close()
        
    start = time()
    
    print('\tGetting Predictions')
    # read in 
    data = pd.read_csv(filedata).dropna(axis=0)
    
    
    #print('\tSplitting data into chunks for multiprocessing')
    #n_cores = 1
    #chunk_size = int(data.shape[0]/n_cores)
    #chunks = [data.iloc[data.index[i:i + chunk_size]] for i in range(0, data.shape[0], chunk_size)]
    
    print('\tStarting Multiprocessing')
    #pool = Pool(n_cores)
    #pool.map(COCO_Results, chunks)
    #pool.close()
    #with Pool(n_cores) as p:
    #    list(tqdm(p.imap(COCO_Results, df_chunking(data,chunk_size)), total=chunk_size))
    
    COCO_Results(data)

    elapsed = time() - start
    print("It took {} hours to run".format(elapsed/3600))    
        
