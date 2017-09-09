
import csv
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

lines=[]
#read lines
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#split lines into train and valication lines
train_lines,validataion_lines = train_test_split(lines,test_size=0.25)

		
images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/'+filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurment)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flattern,Dense

model = Sequential()

#model.add(Flatten(input_shape=(160,320,3))
model.add(Flatten(input_shape=X_train[0].shape))
model.add(Dense(1))
'''
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X_train[0].shape)

#remove top and bottom part of the image as they are not relevant for the training
model.add(Cropping2D(cropping=((70,25),(0,0)))
#1st convolution layer with kernel size 5x5, stride 2x2, depth 24, and with RELU activation
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#2nd convolution layer with kernel size 5x5, stride 2x2, depth 36, and with RELU activation
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#3rd convolution layer with kernel size 5x5, stride 2x2, depth 48, and with RELU activation
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
#4th convolution layer with kernel size 3x3, stride 1, depth 64, and with RELU activation
model.add(Convolution2D(64,3,3,activation="relu"))
#5th convolution layer with kernel size 5x5, stride 1, depth 64, and with RELU activation
model.add(Convolution2D(64,3,3,activation="relu"))
#flatten layer
model.add(Flatten())
#1st fully connected layer with output size 100
model.add(Dense(100))
#2nd fully connected layer with output size 50
model.add(Dense(50))
#3rd fully connected layer with output size 10
model.add(Dense(10))
#final fully connected layer with output size 1 as we have 1 label
model.add(Dense(1))
'''

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,np_epoch=7)

model.save('model.h5')
