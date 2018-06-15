#!/usr/bin/env python

# model.py -- This module implements a flexible model for saliency generation utilizing prednet.

import os
import sys
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from data_utils import PreloadedSequence
from kitti_settings import *
import hickle
from scipy import misc
import pysaliency
from scipy.ndimage import zoom
from scipy.ndimage import filters
from scipy.misc import logsumexp

class PredSaliencyModel:# (pysaliency.SaliencyMapModel):
    
    def __init__(self, weightsfile, prior):
        #self.weights=hickle.load(weightsfile)
        
        # Load PredNet configuration and instantiate a prednet object:
        weights_file = os.path.join(WEIGHTS_DIR, 'prednet_caltech_weights.hdf5')
        json_file = os.path.join(WEIGHTS_DIR, 'prednet_caltech_model.json')
        
        # Load trained model
        f = open(json_file, 'r')
        json_string = f.read()
        f.close()
        train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
        train_model.load_weights(weights_file)
        
        # We have a pretrained model now. 
        layer_config=train_model.layers[1].get_config()
        self.layer="E0"
        layer_config['output_mode']=self.layer
        self.input_shape=list(train_model.layers[0].batch_input_shape[1:])
        # NOTE: We need to remember to set the input shape at 0 to the number of images in the series.
        if prior!="none":
            self.prior=hickle.load(prior)
        else:
            self.prior=None
        self.data_format=layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        self.test_prednet=PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
        
        
    
    # This will return a saliency map based on the input stimulus. 
    def predict(self, stimarray):
        # Determine the actual dimensions of the stimulus array:
        inputdims=(float(stimarray.shape[1]), float(stimarray.shape[2])) # Stimulus array dimensions are nelements, x, y, nchannels. Note that x and y may actually be reversed. (height, then width?)
        # Note that the prior should actually be scaled by the output shape, not the input shape...
        print("Stimulus array shape:")
        print(stimarray.shape)
        if self.prior is not None:
            pshape=self.prior.shape
            print("Self prior shape:")
            print(pshape)
            print("Scaling factor:")
            print(inputdims)
            print((inputdims[0]/pshape[0], inputdims[1]/pshape[1]))
            prior=zoom(self.prior, (inputdims[0]/pshape[0], inputdims[1]/pshape[1]), order=0, mode="nearest")
            # Normalize?
            prior=prior/np.max(prior)
            #prior-=logsumexp(prior)
            print("prior min: %f" % np.min(prior))
            print("prior max: %f" % np.max(prior))
            #return np.arr([prior])
        
        nt=stimarray.shape[0]
        print("nt: %d" % nt)
        batch_size=10#min(nt, 100)
        print("batch_size: %d" % batch_size)
        
        self.input_shape[0]=nt
        #inputs=Input(shape=tuple(self.input_shape))
        print("input shape: ")
        print(self.input_shape)
        #predictions=self.test_prednet(inputs)
        #test_model=Model(inputs=inputs, outputs=predictions)
        # We are experiencing the same problems noticed last year, namely that the precision provided
        # by GPU devices causes a sort of runaway artifacting. As such, we will alter the model to 
        # predict on short, overlapping bursts of data.
        nbursts=stimarray.shape[0]/100
        
        burstarray=[]
        for n in range(nbursts):
            data=stimarray[n*100:n*100+105] #min(n*100+105, n*100+100)]
            nt=data.shape[0]
            batch_size=nt
            test_generator=PreloadedSequence(data, nt, batch_size=batch_size, sequence_start_mode="unique", data_format=self.data_format)
            
            Xtest=test_generator.create_all()
            print("Predicting...")
            #ASSUMPTION: it is safe to set the batch size equal to the number of inputs here since we won't be dealing with many inputs. 
            inputs=Input(shape=tuple(data.shape))
            predictions=self.test_prednet(inputs)
            test_model=Model(inputs=inputs, outputs=predictions)
            predictions=test_model.predict(Xtest, batch_size)
            print("Shape of outputs:")
            print(len(predictions))
            # Now that we have a set of errors, let's do stuff:
            if self.layer=="E0" or self.layer=="Ahat0":
                predictions=predictions[0] # Now the first index will be the image, then channels, then pixels.
                print(len(predictions))
                outlist=[]
                for i in range(predictions.shape[0]): # For each prediction:
                    print(predictions[i].shape)
                    # If it's any other layer, then we want to preserve everything as is.
                    pred=predictions[i].sum(axis=-1) # Sum all channels. axis=0 was summing all elements. Whoops!
                    outlist.append(pred)#filters.gaussian_filter(pred, 4)) # Scale by resized prior distribution.
                predictions=np.array(outlist)
            burstarray.append(predictions)
        data=stimarray[nbursts*100:]
        batch_size=data.shape[0]
        """
        test_generator=PreloadedSequence(data, nt, batch_size=min(batch_size, data.shape[0]), sequence_start_mode="unique", data_format=self.data_format)
        inputs=Input(shape=tuple(data.shape))
        predictions=self.test_prednet(inputs)
        test_model=Model(inputs=inputs, outputs=predictions)
        Xtest=test_generator.create_all()
        print("Predicting...")
        #ASSUMPTION: it is safe to set the batch size equal to the number of inputs here since we won't be dealing with many inputs. 
        predictions=test_model.predict(Xtest, batch_size)
        print("Shape of outputs:")
        print(predictions.shape)
        # Now that we have a set of errors, let's do stuff:
        if self.layer=="E0" or self.layer=="Ahat0":
            predictions=predictions[0] # Now the first index will be the image, then channels, then pixels.
            print(predictions.shape)
            outlist=[]
            for i in range(predictions.shape[0]): # For each prediction:
                print(predictions[i].shape)
                # If it's any other layer, then we want to preserve everything as is.
                pred=predictions[i].sum(axis=-1) # Sum all channels. axis=0 was summing all elements. Whoops!
                outlist.append(pred)#filters.gaussian_filter(pred, 4)) # Scale by resized prior distribution.
            predictions=np.array(outlist)
        burstarray.append(predictions)
        """
        # Now that we have a large array of bursts:
        outarr=np.zeros((stimarray.shape[0], 120, 160))#predictions.shape[1], predictions.shape[2]))
        prevelem=burstarray[0]
        outarr[:prevelem.shape[0]]=prevelem
        for i, elem in enumerate(burstarray[1:]):
            elem[0:5]=prevelem[-5:]
            prevelem=elem
            outarr[i*100:i*100+elem.shape[0]]=elem
        return outarr
        
        

