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
x = np.loadtxt("train_x.csv", delimiter=",")
test = np.loadtxt("test_x.csv", delimiter=",")
t = np.loadtxt("train_y.csv", delimiter=",").astype('int')

# use pkl can read data faster
#x=joblib.load('trainx.pkl')
#test=joblib.load('testx.pkl')

# map result to 40 classes
n_classes = len(set(t))
classes = [i for i in range(0, n_classes)]
l=sorted(set(t))
class_dict1 = dict(zip(l, classes))
class_dict2=dict(zip(classes,l))
y = np.vectorize(class_dict1.get)(t)


# the data, shuffled and split between train and test sets

x_train=x[5000:]
x_test=x[:5000]
y_train=y[5000:]
y_test=y[:5000]

# reshape the data
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    test = test.reshape(test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# normalization 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
test=test.astype('float32')
x_train /= 255.0
x_test /= 255.0
test /=255.0

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build model
model = Sequential()

#block 1
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#block 2
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#block 3
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((4, 4)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#block 4
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# dense-nn, dropout and output layer
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#build the model, test on the validation set
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

# plot all the graph
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

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
		 
model.save('cnn.h5')
