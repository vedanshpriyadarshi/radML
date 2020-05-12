from PIL import Image
from imutils import paths
import numpy as np
import cv2
import os
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import *
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import keras


def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


dataset="desktop/SDR/Data/431"
imagePaths = list(paths.list_images(dataset))
#print(imagePaths)

# getting name of class which is basically folder name
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
#print(classNames)
#load images

def load(imagePaths):
    # initilize the list of features and labels
    data = []
    labels = []

    # loop over the input images
    for(i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label
        # assuming that our path has the following format
        image = Image.open(imagePath) 
        label = imagePath.split(os.path.sep)[-2]
        
        coords=(103.5,16.25,972.5,663)
        image=image.crop(coords)
        #image=cv2.resize(image,(224,224))
        #image.save(image)
        #image.show()
        # change image to array
        image=image.convert("RGB")
        image = img_to_array(image, data_format=None)
        image=cv2.resize(image,(224,224))
        
  
        data.append(image)
        labels.append(label)
    
    # return as numpy array
    return(np.array(data), np.array(labels))


(data,labels)=load(imagePaths)
#data[0].show()
print(labels)


data = data.astype("float") / 255.0

y=[]
#print(labels)
for i in range(len(labels)):
    # if (labels[i]=="null"):
    #     y.append([1,0,0])
    if(labels[i]=="null"):
        y.append([1,0])
    elif(labels[i]=="water"):
        y.append([0,1])   

print(y)
y=np.array(y)



#labels=LabelBinarizer().fit_transform(labels)
#print(labels)

#Split into training and test
trainX, testX, trainY, testY = train_test_split(data, y, test_size=0.1, random_state=42)
#Splitting in training and validation from training set
#trainX, valX, trainY, valY   = train_test_split(trainX, trainY, test_size=0.2, random_state=42)


# convert the labels from integers to vectors
# trainY = LabelBinarizer().fit_transform(trainY)

# testY = LabelBinarizer().fit_transform(testY)
# valY = LabelBinarizer().fit_transform(valY)

# pyplot.imshow(trainX[100])
# print(trainX.shape)
# pyplot.show()
# print(trainY[100])

model = keras.models.Sequential()

# model.add(Conv2D(32, (3, 3), padding="same",input_shape=(128,128,3)))
# model.add(Activation("relu"))
# #model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))



#model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same', input_shape=(224,224,3)))

model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

# model.add(Dropout(0.4))



model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))





model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))


model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

# 		# (CONV => RELU) * 2 => POOL
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(Conv2D(32, (3, 3), padding="same"))
# model.add(Activation("relu"))
# #model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# 		# (CONV => RELU) * 2 => POOL
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# #model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
# #model.add(BatchNormalization())


#model.add(Flatten())
model.add(Dense(2))
model.add(Activation("softmax"))
#model.add(BatchNormalization())


# model.add(Conv2D(32, kernel_size=5, strides=2, activation='relu', input_shape=(28, 28, 3)))
# model.add(Dropout(0.3))
# model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))     
# model.add(Flatten())

# model.add(Dense(128, activation='relu'))
# model.add(Dense(18, activation='softmax'))   # Final Layer using Softmax

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',top_1_accuracy])

model.fit(trainX,trainY, batch_size=32, nb_epoch=10, verbose=1,validation_split=0.1)

score = model.evaluate(testX, testY, batch_size=16)

print(score)
model.save('all.h5')

pred=model.predict(testX)

print(pred)
print(testY)