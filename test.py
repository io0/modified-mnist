# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:57:39 2017

@author: Marley
"""
import pandas
import numpy as np
from math import exp, floor, log

if (firstrun):
    limit = 10000
    a = pandas.read_csv("train_x_shuffled.csv", header=None, nrows=limit)
    x = a.as_matrix()
    #x = x.reshape(-1, 4096)
    x /= 255.0
    
    y = np.loadtxt("train_y_shuffled.csv", delimiter=",")
    y = y[:limit]
    n_classes = len(set(y))
    classes = [i for i in range(0, n_classes)]
    class_dict = dict(zip(list(set(y)), classes))
    y = np.vectorize(class_dict.get)(y)
    y = np.eye(n_classes)[y] # one-hot vector
    
    init_wi = np.loadtxt("wi.csv", delimiter = ",")
    init_wi2 = np.loadtxt("wi2.csv", delimiter = ",")
    init_wo = np.loadtxt("wo.csv", delimiter = ",")
    
    g_sigmoid = []
    for i in range(0, 200):                                                         # Store sigmoid to save time
        n = (i / 20.0) - 5.0;
        g_sigmoid.append(1.0 / (1.0 + exp(-1.0 * n)))
    
    '''Some helper functions for the network'''
    constrain = lambda n, minn, maxn: max(min(maxn, n), minn)
    def sigmoid(x):
        return g_sigmoid[constrain(int(floor((x+5.0)*20.0)), 0, 199)]
    def t_shape(array):                             # Transpose and reshape
        if array.ndim < 2:
            return array.reshape(-1,1)
        else:
            return array.transpose()
    def r_shape(array):
        if array.ndim < 2:
            return array.reshape(1,-1)
        else:
            return array
def softmax(array):
    ps = np.exp(array)
    return ps/t_shape(np.sum(ps,axis=-1))

class Network():
    def __init__(self, input, output, hidden_size, num_layers, batch_size=250, learning_rate=0.002, num_folds = 2, epochs=2, init_wi = [], init_wi2 = [], init_wo = []):
        self.input = np.insert(input, 0, 1, axis = -1)
        self.output = output
        self.input_size = input.shape[-1] # 4096 inputs
        self.hidden_size = hidden_size
        self.output_size = output.shape[-1] # 40 outputs
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_folds = num_folds
        self.partition = int(len(input)/num_folds)
        self.epochs = epochs
        self.wi = init_wi
        self.wi2 = init_wi2
        self.wo = init_wo
        
        if not len(init_wi):
            input_range = 1.0 / self.input_size ** (1/2)
            self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input_size + 1, self.hidden_size))
            np.savetxt("wi.csv", self.wi, delimiter=",")
        if not len(init_wi2):
            input_range = 1.0 / self.hidden_size ** (1/2)
            self.wi2 = np.random.normal(loc = 0, scale = input_range, size = (self.hidden_size + 1, self.hidden_size))
            np.savetxt("wi2.csv", self.wi2, delimiter=",")
        if not len(init_wo):
            hidden_range = 1.0 / self.hidden_size ** (1/2)
            self.wo = np.random.normal(loc = 0, scale = hidden_range, size = (self.hidden_size + 1, self.output_size))
            np.savetxt("wo.csv", self.wo, delimiter=",")
            
        self.training_loop()
        self.test_debug()
    def training_loop(self):
        for i in range(self.num_folds):
            self.run_training() 
            self.input = np.concatenate((self.input[self.partition:], self.input[:self.partition]))                          # Rotate list to obtain different partitioning
            self.output = np.concatenate((self.output[self.partition:], self.output[:self.partition]))
    def run_training(self):
        for i in range (self.epochs):
            b = 0
            while (b < len(self.input) - self.partition):
                input = self.input[b:b + self.batch_size]
                output = self.output[b:b + self.batch_size]
                self.forward_pass(input)
                self.back_propagate(input, output)
                if (b%200 == 0):
                    self.print_loss(output)
                b += self.batch_size
            self.run_validation()

    def forward_pass(self, input):
        self.h_output = np.matmul(input, self.wi)
        self.h_output = np.vectorize(sigmoid)(self.h_output)
        self.h_output = np.insert(self.h_output, 0, 1, axis = -1)
        #print("h_output:",self.h_output.shape)
        if (self.num_layers == 2):
            self.h2_output = np.matmul(self.h_output, self.wi2)
            self.h2_output = np.vectorize(sigmoid)(self.h2_output)
            self.h2_output = np.insert(self.h2_output,0,1,axis = -1)
            self.o_output = np.matmul(self.h2_output,self.wo)
        else:
            self.o_output = np.matmul(self.h2_output, self.wo)
        
        self.o_output = softmax(self.o_output)
        #print("ooutput:",self.o_output.shape)
        
    def back_propagate(self, input, output):
        o_delta = self.o_output - output                                # - (yi - oj) = (oj - yi)
        #print("odelta:",o_delta.shape)
        #print("output:",output.shape)
        o_update = np.matmul(t_shape(self.h_output),r_shape(o_delta))   # h_output is the vector of inputs to the output layer
        #print("oupdate:",o_update.shape)
        
        if (self.num_layers == 2):
            h2_delta = np.matmul(o_delta, t_shape(self.wo)) * self.h2_output * (1 - self.h2_output)
            h2_update = np.matmul(t_shape(self.h_output), r_shape(h2_delta[:,1:]))
            
            h_delta = np.matmul(h2_delta[:,1:], t_shape(self.wi2)) * self.h_output * (1 - self.h_output)
            h_update = np.matmul(t_shape(input), r_shape(h_delta[:,1:]))
        else:
            h_delta = np.matmul(o_delta, t_shape(self.wo)) * self.h_output * (1 - self.h_output)
            h_update = np.matmul(t_shape(input), r_shape(h_delta[:,1:]))
        #print(h_update.shape)
        
        self.wo = self.wo - self.learning_rate * o_update
        self.wi = self.wi - self.learning_rate * h_update
        if self.num_layers == 2:
            self.wi2 = self.wi2 - self.learning_rate * h2_update
    def run_validation(self):
        correct = 0
        for i in range (len(self.input) - self.partition,len(self.input)):
            self.forward_pass(self.input[i])
            if (np.argmax(self.o_output) == np.argmax(self.output[i])):
                #print(np.argmax(self.output[i]))
                correct += 1
        print(correct/self.partition)
    

    def print_loss(self,output):
        print (np.argmax(self.o_output, axis = -1))
        print (np.argmax(output, axis = -1))
        E = output * np.log(self.o_output)
        #E = output * np.vectorize(log)(self.o_output) + (1 - output)*np.vectorize(log)(1 - self.o_output)
        print("E", E.shape)
        E = 0 - np.sum(E)/self.batch_size
        print(E)
        
    def test_debug(self):
        print("input shape:")
        print(self.input.shape)
        print("output shape:")
        print(self.output.shape)
        print("h_output:")
        print(self.h_output.shape)
        print("o_output:")
        print(self.o_output.shape)
        
nn = Network(x, y, 2048, 2, init_wi = init_wi, init_wi2 = init_wi2, init_wo = init_wo)
'''
reader = csv.reader(open("test.csv", "rb"), delimiter=",")
x = list(reader)
result = numpy.array(x).astype("float")

import numpy   as np 
import scipy.misc # to visualize only  
x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
y = np.loadtxt("train_y.csv", delimiter=",") 
x = x.reshape(-1, 64, 64) # reshape 
y = y.reshape(-1, 1) 
scipy.misc.imshow(x[0]) # to visualize only 
'''


