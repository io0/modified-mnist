# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:57:39 2017

@author: Marley
"""
import pandas
import numpy as np
from math import exp, floor, log

if (firstrun):
    limit = 1000
    a = pandas.read_csv("train_x.csv", nrows=limit)
    x = a.as_matrix()
    x = x.reshape(-1, 4096)
    x /= 255.0
    
    y = np.loadtxt("train_y.csv", delimiter=",")
    y = y[:limit]
    n_classes = len(set(y))
    classes = [i for i in range(0, n_classes)]
    class_dict = dict(zip(list(set(y)), classes))
    y = np.vectorize(class_dict.get)(y)
    y = np.eye(n_classes)[y] # one-hot vector
    
    init_wi = np.loadtxt("wi.csv", delimiter = ",")
    init_wo = np.loadtxt("wo.csv", delimiter = ",")
    
    g_sigmoid = []
    for i in range(0, 200):                                                         # Store sigmoid to save time
        n = (i / 20.0) - 5.0;
        g_sigmoid.append(1.0 / (1.0 + exp(-1.0 * n)))

constrain = lambda n, minn, maxn: max(min(maxn, n), minn)
def sigmoid(x):
    return g_sigmoid[constrain(int(floor((x+5.0)*20.0)), 0, 199)]

class Network():
    def __init__(self, input, output, hidden_size, init_wi = [], init_wo = []):
        self.input = np.insert(input, 0, 1, axis = -1)
        self.output = output
        self.input_size = input.shape[-1] # 4096 inputs
        self.hidden_size = hidden_size
        self.output_size = output.shape[-1] # 40 outputs
        self.wi = init_wi
        self.wo = init_wo
        
        if not len(init_wi):
            input_range = 1.0 / self.input_size ** (1/2)
            self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input_size + 1, self.hidden_size))
            np.savetxt("wi.csv", self.wi, delimiter=",")
        if not len(init_wo):
            hidden_range = 1.0 / self.hidden_size ** (1/2)
            self.wo = np.random.normal(loc = 0, scale = hidden_range, size = (self.hidden_size + 1, self.output_size))
            np.savetxt("wo.csv", self.wo, delimiter=",")
        self.run_training()
        self.test_debug()
    def run_training(self):
        for i in range (0, 100):
            self.forward_pass(self.input[i])
            self.back_propagate(self.input[i], self.output[i], 0.0002)
            self.print_error(self.output[i])
        
    def forward_pass(self, input):
        self.h_output = np.matmul(input, self.wi)
        self.h_output = np.vectorize(sigmoid)(self.h_output)
        self.h_output = np.insert(self.h_output, 0, 1, axis = -1)
        self.o_output = np.matmul(self.h_output, self.wo)
        self.o_output = np.vectorize(sigmoid)(self.o_output)
        #print(self.o_output)
    def back_propagate(self, input, output, learning_rate):
        o_delta = self.o_output - output                                # - (yi - oj) = (oj - yi)
        o_update = np.asarray([o_delta * x for x in self.h_output])   # h_output is the vector of inputs to the output layer
        '''o_delta = self.o_output - output                                # - (yi - oj)
        o_delta = np.tile(o_delta, (self.hidden_size + 1, 1))
        o_update = o_delta * self.h_output.reshape(-1,1)'''
        
        h_delta = np.matmul(self.wo, o_delta) * self.h_output * (1 - self.h_output)
        h_update = np.asarray([h_delta[:-1] * x for x in input])
        
        self.wo = self.wo - learning_rate * o_update
        self.wi = self.wi - learning_rate * h_update
        
    def print_error(self,output):
        print (np.argmax(self.o_output))
        print (np.argmax(output))
        E = output * np.vectorize(log)(self.o_output) + (1 - output)*np.vectorize(log)(1 - self.o_output)
        E = 0 - np.sum(E)/40.0
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
        
nn = Network(x, y, 2048, init_wi = init_wi, init_wo = init_wo)
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


