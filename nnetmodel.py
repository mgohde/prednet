#!/usr/bin/env python
# This implements a very small neural network that is trained based on
# the values present within prednet's error layers.
import keras
import hickle
import os
import sys
import numpy as np
from scipy import misc

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten
from keras import optimizers

# This model attempts to learn based on the features present in Prednet's third error layer.
# (1, 1749, 384, 16, 20)
# These appear to be:
# (unused, imageindex, dimension, x, y)

class nnetmodel:
    def __init__(self, inputshape, outputshape):
        # Deep Gaze II utilizes four layers of 1x1 convolutions. We will use this as a baseline until 
        # network optimization starts to come into play. 
        self.model=Sequential()
        # As I recall, error layers such as E3 have fairly oddly dimensioned features.
        # I *think* we want one filter per element in the input.
        # TODO: consider using nonlinear activations within the convolution layers as per Deep Gaze II
        nfilters=nfilters=inputshape[2]#inputshape[2]*inputshape[3]*inputshape[4]
        #self.model.add(Conv2D(64, 2))
        self.model.add(Conv2D(64, 1, input_shape=inputshape[2:], data_format="channels_first"))
        #self.model.add(Conv2D(nfilters, 1, data_format="channels_first"))
        self.model.add(Flatten())
        #self.model.add(Flatten(input_shape=inputshape[2:]))
        #self.model.add(Dense(40*30, activation="relu"))
        #self.model.add(Dense(40*30, activation="softmax"))
        #self.model.add(Dense(80*60, activation="relu"))
        self.model.add(Dense(80*60, activation='softmax'))
        self.model.add(Dense(80*60, activation='sigmoid'))
        # TODO: set a loss function!
        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer="adadelta", loss="binary_crossentropy") #loss='mean_squared_error', optimizer=sgd)

    def train(self, inputdata, refmapdata, nepochs):
        print("Attempting to train the model!")
        self.model.fit(x=inputdata[0], y=refmapdata, epochs=nepochs)
        
    
    def loadweights(self, weightsfile):
        self.model.load_weights(weightsfile)
    
    def saveweights(self, weightsfile):
        self.model.save_weights(weightsfile)
        
        # Fun fact: it's possible to load weights into an architecture for which there are named layers in common.
    
    def predict(self, inputdata):
        return self.model.predict(inputdata[0])
