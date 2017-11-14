#shuran zheng 260627203

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np 
import csv

#x=joblib.load('trainx.pkl')
#test=joblib.load('testx.pkl')

# read and reshape the data
x = np.loadtxt("train_x.csv", delimiter=",")
test = np.loadtxt("test_x.csv", delimiter=",")
t = np.loadtxt("train_y.csv", delimiter=",").astype('int')
t=t.reshape(-1)
x=x.reshape(-1,64*64)
test=test.reshape(-1,64*64)

# map the result to 40 classes
n_classes = len(set(t))
classes = [i for i in range(0, n_classes)]
l=sorted(set(t))
class_dict1 = dict(zip(l, classes))
class_dict2 = dict(zip(classes, l))
y = np.vectorize(class_dict1.get)(t)

# the data, shuffled and split between train and test sets
x_train=x[:45000]
x_test=x[45000:]
y_train=y[:45000]
y_test=y[45000:]

# normalize the data into 0 to 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
test=test.astype('float32')
x_train /= 255.0
x_test /= 255.0
test /=255.0

# build logistic regression model
logisticRegr=LogisticRegression()
logisticRegr.fit(x_train,y_train)

#predit the test set
predictions=logisticRegr.predict(x_test)

# map 40 classses to the real digits
predictions=np.vectorize(class_dict2.get)(predictions)

# write to the library
with open("logistic.csv",'w') as output:
     output.write("Id,Label")
     output.write("\n")
     for i in range(len(predictions)):
         output.write(str(i+1))
         output.write(",")
         output.write(str(predictions[i]))
         output.write("\n")

# the score of validtion set
score=logisticRegr.score(x_test,y_test)
print(score)

