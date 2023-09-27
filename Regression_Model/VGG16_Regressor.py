import numpy as np
import pandas as pd
import os
import tensorflow as tf
print('Tensorflow version : {}'.format(tf.__version__))
print('GPU : {}'.format(tf.config.list_physical_devices('GPU')))
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Reshape, Activation, Conv2D, Input, MaxPool2D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

#Directory for COCO blurred dataset
parent_dir = '/home/varelal/Documents/COCO_blurred_V1/'

#Directory for each dataset
train_dir = parent_dir + 'Train/'
test_dir = parent_dir + 'Test/'
val_dir = parent_dir + 'Validate/'

#Trained labels
train_labels = parent_dir + 'train_dataset.csv'
test_labels = parent_dir + 'test_dataset.csv'
val_labels = parent_dir +'val_dataset.csv'

#Where to save log data
csv_logger_dir = 'log/log_V20.csv'

#Save weights filename
weights_dir = 'Weights_V20.h5'

IMAGE_H, IMAGE_W = 224, 224
CHANNELS = 3
TRAIN_BATCH_SIZE = 50
VAL_BATCH_SIZE = 32
EPOCHS = 50

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
                    
            else:
                print("Error couldn't load:", self.directory + ID['filename'])
                
            
        return X, y
  
# Model  
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



#Read in csv and normalize angle and length
train_pd = pd.read_csv(train_labels)
train_pd['angle'] = (train_pd['angle'] - min(train_pd['angle'])) / (max(train_pd['angle']) - min(train_pd['angle']))
train_pd['length'] = (train_pd['length'] - min(train_pd['length'])) / (max(train_pd['length']) - min(train_pd['length']))

#Read in validation csv and normalize angle and length
val_pd = pd.read_csv(val_labels)
val_pd['angle'] = (val_pd['angle'] - min(val_pd['angle'])) / (max(val_pd['angle']) - min(val_pd['angle']))
val_pd['length'] = (val_pd['length'] - min(val_pd['length'])) / (max(val_pd['length']) - min(val_pd['length']))


#Generators
train_generator = DataGenerator(train_dir,
                                train_pd,
                                batch_size=TRAIN_BATCH_SIZE)

val_generator = DataGenerator(val_dir,
                              val_pd,
                              batch_size=VAL_BATCH_SIZE)
    
#Callbacks
checkpoint = ModelCheckpoint(filepath=weights_dir)

earlystop = EarlyStopping(monitor='val_loss',
                         min_delta=10e-12,
                         patience=5,
                         mode='min',
                         restore_best_weights=True)

csv_logger = CSVLogger(csv_logger_dir, append=True, separator=';')


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.135, epsilon=0.1),
              loss=tf.keras.losses.MeanSquaredError())

model.fit(train_generator,
                validation_data = val_generator,
                epochs = EPOCHS,
                callbacks=[checkpoint, earlystop, csv_logger])
