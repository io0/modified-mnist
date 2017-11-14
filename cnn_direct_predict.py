#shuran zheng 260627203
from __future__ import print_function
import keras
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Convolution2D, ZeroPadding2D
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import optimizers

batch_size = 256
num_classes = 40
epochs = 20
img_rows, img_cols = 64, 64

# read data
test = np.loadtxt("test_x.csv", delimiter=",")
t = np.loadtxt("train_y.csv", delimiter=",").astype('int')
#test=joblib.load('testx.pkl')

# map result to 40 classes
n_classes = len(set(t))
classes = [i for i in range(0, n_classes)]
l=sorted(set(t))
class_dict1 = dict(zip(l, classes))
class_dict2=dict(zip(classes,l))
y = np.vectorize(class_dict1.get)(t)

# reshape the data
if K.image_data_format() == 'channels_first':
    test = test.reshape(test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# normalization 
test=test.astype('float32')
test /=255.0

# load model
model = load_model('cnn.h5')

# map the 40 classes back to result
result = model.predict_classes(test)
predictions = np.vectorize(class_dict2.get)(result)
		 
with open("cnn.csv",'w') as output:
     output.write("Id,Label")
     output.write("\n")
     for i in range(len(predictions)):
         output.write(str(i+1))
         output.write(",")
         output.write(str(predictions[i]))
         output.write("\n")
		 
