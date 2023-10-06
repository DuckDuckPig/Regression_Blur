import numpy as np
import pandas as pd
import os
import progressbar
import tensorflow as tf
print('Tensorflow version : {}'.format(tf.__version__))
print('GPU : {}'.format(tf.config.list_physical_devices('GPU')))
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Reshape, Activation, Conv2D, Input, MaxPool2D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input, Model
import glob

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


######## Parameters ##########################
parent_dir = '/home/varelal/Documents/COCO_blurred_V1/'

train_dir = parent_dir + 'Train/'
test_dir = parent_dir + 'Test/'
val_dir = parent_dir + 'Validate/'
test_dir = test_dir 

train_labels = parent_dir + 'train_dataset.csv'
test_labels = parent_dir + 'test_dataset.csv'
val_labels = parent_dir +'val_dataset.csv'
test_labels = test_labels
IMAGE_H, IMAGE_W = 224, 224
CHANNELS = 3
TRAIN_BATCH_SIZE = 10
VAL_BATCH_SIZE = 10
EPOCHS = 100

weights_filenames = ['weights.h5']


#Version 2 where we input all dataframe as one
class DataGenerator(keras.utils.Sequence):
    def __init__(self, directory, dataframe, len_y= 2, batch_size=32, dim=(224,224), n_channels=3, shuffle=True):
        self.dim = dim
        self.directory = directory
        self.batch_size = batch_size
        #self.y_length = y_length
        #self.y_angle = y_angle
        #self.list_IDs = list_IDs
        self.dataframe = dataframe
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.len_y = len_y
        self.on_epoch_end()

    def on_epoch_end(self):
        self.index = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):

        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        #Generates one batch of data
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        
        #find list of IDs
        list_IDs_temp = [self.dataframe.loc[k] for k in index]
        
        #enerate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
        
    def random_crop(self, img, random_crop_size):

        height, width = img.shape[0], img.shape[1]      
        dy, dx = random_crop_size
        
        if height >= dy and width >= dx:
            x = np.random.randint(0,width - dx + 1)
            y = np.random.randint(0,height - dy + 1)
        
            return img[y:(y+dy), x:(x+dx),:]/255.0

        else:
            return 0
        
    def __data_generation(self, list_IDs_temp):
        #Generates data containing batch_size samples X: (n_samples, *dim, n_channels)
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.len_y), dtype=float)
        #print("listID:", type(list_IDs_temp))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #print("ID:",ID['filename'], ID['angle'], ID['length'])
            #print("i:",i)
            if os.path.exists(self.directory + ID['filename']):
                #store sample
                img = keras.preprocessing.image.load_img(self.directory + ID['filename'])
                img = keras.preprocessing.image.img_to_array(img)
                height, width = img.shape[0], img.shape[1]                
                dy, dx = self.dim
                
                if height >= dy and width >= dx:
                    X[i,] = self.random_crop(img, random_crop_size=self.dim)
                
                    #store class
                    y[i,] = [ID['length'],ID['angle']]
            
        return X, y
    
    
def random_crop(img, random_crop_size):

    height, width = img.shape[0], img.shape[1]      
    dy, dx = random_crop_size
        
    if height >= dy and width >= dx:
        #x = np.random.randint(0,width - dx + 1)
        #y = np.random.randint(0,height - dy + 1)
        cx = int(width//2)
        cy = int(height//2)

        #return img[y:(y+dy), x:(x+dx),:]/255.0
        return img[int(cy-112):int(cy+112),int(cx-112):int(cx+112),:]/255.0
    
#%% VGG16
model = Sequential()

model.add(Conv2D(input_shape=(IMAGE_H, IMAGE_W, CHANNELS), filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=2, activation='sigmoid')) 

#Grab all weights in directory
#weights_filenames = sorted(glob.glob('*.h5'))
print(weights_filenames)


R2_angle = []
R2_length = []


for weight_filename in weights_filenames:
    test_pd = pd.read_csv(test_labels)
    angle_min = min(test_pd['angle'])
    angle_max = max(test_pd['angle'])
    length_min = min(test_pd['length'])
    length_max = max(test_pd['length'])

    test_pd['Pred_angle'] = np.nan
    test_pd['Pred_length'] = np.nan


    model.load_weights(weight_filename)
    print("Results for:", weight_filename)

    filenames = np.asarray(test_pd['filename'])


    bar = progressbar.ProgressBar(maxval=len(filenames), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i, filename in enumerate(filenames):
        #read image
        img = keras.preprocessing.image.load_img(test_dir + filename)
        img = keras.preprocessing.image.img_to_array(img)

        #preprocessing
        height, width = img.shape[0], img.shape[1]                
        dy, dx = (224,224)

        if height >= dy and width >= dx:
            img_crop = np.expand_dims(random_crop(img, (224,224)),0)
        else:
            continue

        #detection
        pred = model.predict(img_crop)
        #denormalize
        length = (length_max - length_min)*pred[0,0] + length_min
        angle = (angle_max - angle_min)*pred[0,1] + angle_min


        test_pd.loc[test_pd['filename']==filename,'Pred_angle'] = angle
        test_pd.loc[test_pd['filename']==filename,'Pred_length'] = length

        bar.update(i + 1)



    test_pd1 = test_pd.dropna(axis=0)

    if not test_pd1.empty:
        coeff_angle = r2_score(np.asarray(test_pd1['angle']),np.asarray(test_pd1['Pred_angle']))
        coeff_length = r2_score(np.asarray(test_pd1['length']),np.asarray(test_pd1['Pred_length']))

        #R2 Angle
        plt.figure(figsize=(10,10))
        plt.scatter(test_pd1['angle'],test_pd1['Pred_angle'])


        x = np.unique(test_pd1['angle'])
        m, b = np.polyfit(test_pd1['angle'],test_pd1['Pred_angle'],1)
        plt.plot(x, m*x + b, 'red')
        plt.text(-90,90,'R^2={}'.format(coeff_angle),fontsize='x-large')
        plt.title('Angle')
        plt.xlabel('Actual')
        plt.ylabel('Prediction')

        plt.savefig('Results/' + weight_filename.split('.')[0] + '_angle.png')
        #plt.savefig('Results/Validate/' + os.path.basename(weight_filename).split('_')[1] + '_angle.png')

        #R2 Length
        plt.figure(figsize=(10,10))
        plt.scatter(test_pd1['length'],test_pd1['Pred_length'])


        x = np.unique(test_pd1['length'])
        m, b = np.polyfit(test_pd1['length'],test_pd1['Pred_length'],1)
        plt.plot(x, m*x + b, 'red')
        plt.text(0,85,'R^2={}'.format(coeff_length),fontsize='x-large')
        plt.title('Length')
        plt.xlabel('Actual')
        plt.ylabel('Prediction')

        plt.savefig('Results/' + weight_filename.split('.')[0] + '_length.png')
        #plt.savefig('Results/Validate/' + os.path.basename(weight_filename).split('_')[1] + '_length.png')

        # R2 collection
        R2_angle.append(coeff_angle)
        R2_length.append(coeff_length)

    else:
        # R2 collection (0)
        print("All predicted values Nan")
        R2_angle.append(0)
        R2_length.append(0)

"""
    plt.figure(figsize=(10,10))
    plt.plot(np.asarray(range(1,len(R2_angle)+1)), R2_angle)
    plt.title("R2 Angle between Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("R^2 of Angle")
    plt.savefig('Results/' + weights_dir + '/R2_angle_vs_Epoch.png')
    #plt.savefig('Results/Validate/R2_angle_vs_Epoch.png')

    plt.figure(figsize=(10,10))
    plt.plot(np.asarray(range(1,len(R2_length)+1)), R2_length)
    plt.title("R2 Length between Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("R^2 of Length")
    plt.savefig('Results/' + weights_dir + '/R2_length_vs_Epoch.png')
    #plt.savefig('Results/Validate/R2_length_vs_Epoch.png')
"""
