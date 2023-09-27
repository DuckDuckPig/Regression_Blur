# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:33:12 2020

@author: varel
"""
import numpy as np
from scipy.signal import convolve2d
from skimage.draw import line
import matplotlib.pyplot as plt


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
    
    return(K,n_r, n_angle)

def Motion_blur(I,r,angle):
    if len(I.shape) < 3:
        I = np.stack((I,I,I), axis=2)
        
    
    K, n_r, n_angle = Motion_blur_kernel(r,angle)
    I_r = convolve2d(I[:,:,0],K, mode='same')
    I_g = convolve2d(I[:,:,1],K,mode='same')
    I_b = convolve2d(I[:,:,2],K,mode='same')
    I_blurr = np.stack((I_r,I_g,I_b), axis=2)
    return(I_blurr.astype('uint8'),n_r, n_angle)
