from keras import models, layers
import os
import cv2
import numpy as np

TRAIN_DIR = "/content/American-Sign-Language/Gestures"

num_classes=24
IMG_SIZE = 100
def vectorize_data(TRAIN_DIR):
    result = []
    labels = []
    for label in os.listdir(TRAIN_DIR):
        path=""
        path=os.path.join(TRAIN_DIR, label)
        for img in os.listdir(path):
            path2=""
            path2 = os.path.join(path, img)
            i = cv2.imread(path2)
            #i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            
            i = cv2.resize(cv2.imread(path2, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            
            
            result.append(i)
            labels.append(label)
    
    return result, labels

x, y =vectorize_data(TRAIN_DIR)
x_train = np.array(x)
y_train = np.array(y)

x_train = np.expand_dims(x_train, axis=-1)
x_train.shape

from keras.utils.np_utils import to_categorical
dictonary = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'O':13, 'P':14, 
            'Q':15, 'R':16, 'S':17, 'T':18, 'U':19, 'V':20, 'W':21, 'X':22, 'Y':23}
num_classes=24
keys, inv = np.unique(y_train, return_inverse=True)
vals = np.array([dictonary[key] for key in keys])
y_train_new = vals[inv]
y_train_new_cat = to_categorical(y_train_new, num_classes)


'''
keys, inv = np.unique(y_test, return_inverse=True)
vals = np.array([dictonary[key] for key in keys])
y_test_new = vals[inv]
y_test_new_cat = to_categorical(y_test_new,num_classes=24)
'''
# SHUFFLE
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]
x_new,y_new = unison_shuffled_copies(x_train,y_train_new_cat)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
IMG_SIZE = 100

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
IMG_SIZE = 50

num_classes = 26
model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(IMG_SIZE, IMG_SIZE, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))
model.compile(optimizer=SGD(0.0001),loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_new, y_new, epochs = 20, validation_split = 0.1, shuffle = True, batch_size = 500)
model.save("cnn_model2.h5")