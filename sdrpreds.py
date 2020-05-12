from PIL import Image
from imutils import paths
import numpy as np
import cv2
from keras.models import load_model
import os
from keras import models
import matplotlib
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot
import tensorflow as tf
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
#print(labels)


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


y=np.array(y)


def encode(predictions):
    newOut = np.ndarray((len(predictions), len(predictions[0])))
    for i in range(len(predictions)):
        row = predictions[i]
        m = max(row)
        rowAlt = [e for e in row if e != m]
        tx = max(rowAlt)
        for j in range(len(predictions[0])):
            if row[j] == m:
                newOut[i][j] = 1
            else:
                newOut[i][j] = 0
    return newOut


def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


custom={'top_1_accuracy':top_1_accuracy}
model=load_model("all.h5",custom)

#tf.keras.utils.plot_model(model, to_file='saveit.png', show_shapes=True, show_layer_names=True)

#labels=LabelBinarizer().fit_transform(labels)
#print(labels)

#Split into training and test
trainX, testX, trainY, testY = train_test_split(data, y, test_size=0.1, random_state=42)
#Splitting in training and validation from training set
score = model.evaluate(testX, testY, batch_size=16)
pred=model.predict(testX)
pred=encode(pred)


from sklearn.metrics import confusion_matrix
conf=confusion_matrix(testY.argmax(axis=1),pred.argmax(axis=1))





# import seaborn as sns

# plt.figure(figsize=(10, 8))
# sns.heatmap(conf, cmap='YlGnBu', annot=True, square=True, fmt="d")


# plt.ylabel('True Label', fontsize=14)
# plt.xlabel('Predicted Label', fontsize=14)
# plt.title("confusion matrix")
# plt.show()



layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
img = np.expand_dims(testX[2], axis=0)
activations = activation_model.predict(img) 
first_layer_activation = activations[1]
print(first_layer_activation.shape)



plt.matshow(first_layer_activation[0, :, :, 29], cmap='viridis')
plt.show()

layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
