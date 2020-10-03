#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.image import imread
import sys, shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('loss') <= 9):
			self.model.stop_training=True

callbacks = myCallback()
#os.chdir('..')
main_path = os.getcwd()
data_path = main_path + '\\datasets\\face_age'
def get_data():
    
    ages = os.listdir(data_path)
    X = []
    Y = []
    for age in tqdm(ages):
        ageint = int(age)

        new_folder = data_path + "\\{0}".format(age)
        photos = os.listdir(new_folder)
        for photo in photos:
            img = imread("{0}/{1}".format(new_folder, photo))
        
            X.append(img)
            Y.append(ageint)
    X = np.array(X, dtype=np.float16)
    Y = np.array(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)





def nn_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='relu')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

def train_model():
    (X_train, X_test, y_train, y_test) = get_data()
    model = nn_model()
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=900, callbacks=[callbacks])
    model.save_weights('my_weights')
    model.save('my_model')

def return_age(frame):
    #frame = imread(frame, 0)/255
    frame = tf.expand_dims(frame, 0)
    new_model = nn_model()
    
    new_model.load_weights(main_path + '\\my_weights')
    p = new_model.predict(frame)
    return (str(int(np.round(p)[0][0])-3) + " - " + str(int(np.round(p)[0][0])+3))






