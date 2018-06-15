#!/usr/bin/env python
# runmodel.py -- A much cleaner framework with which one may run the saliency extraction model constructed here.
# Why does this exist? It's clear that my current implementations of the model and related code are hacky at best.
# As such, a cleaner, better thought out implementation utilizing language features with an emphasis on readability
# should enable me to perform better in testing and developing this project. 
import model
import os
import sys
import hickle
from scipy import misc
from scipy.ndimage import filters
from kitti_settings import *
import numpy as np

def softmax(im):
    return np.exp(im) / np.sum(np.exp(im), axis=0)


def main(args):
    if len(args)!=5:
        print("Usage: runmodel.py infile prior [-o/-d] [outfile/outdir]")
        print("If -o is specified, then all predictions are dumped to a hickle.")
        print("Else, all predictions are dumped to images in the directory specified.")
        print("If 'prior' is 'extract', then this program will instead extract an existing predictions hickle into the specified output directory.")
        return
    
    infile=args[1]
    prior=args[2]
    outopt=args[3]
    outfile=args[4]
    
    if prior=="extract":
        predictions=hickle.load(infile)
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        
        for p in range(predictions.shape[0]):
            #arr=predictions[p]-np.min(predictions[p])
            #arr=np.abs(predictions[p])
            #arr=np.abs(predictions[p])
            # Scale the array:
            #print(arr.shape)
            #arr=arr/arr.max()
            #arr=arr*255.0
            im = misc.imresize(predictions[p], (480, 640)).astype("float32")
            im=im/im.max()
            im=filters.gaussian_filter(im, 8).astype("float32")
            # We should probably now normalize the image:
            #im=im/np.max(im)
            #im=softmax(im)
            print(im.max())
            print(im.min())
            im[im<im.mean()]=0
            im=im/im.max()
            # Todo: consider thresholding the image to remove all noticable large regions of activation.
            #im=softmax(im)
            print(im.max())
            print(im.min())
            misc.imsave(os.path.join(outfile, "%d.png" % p), im)
            print("Done")
        return
    
    indata=hickle.load(infile)
    m=model.PredSaliencyModel(None, prior) # Prior is passed as a filename and the first argument has yet to be defined.
    
    # Todo: figure out if "indata" is actually formatted correctly for this application.
    predictions=m.predict(indata)
    
    if outopt=='-o':
        print('predictions.shape:')
        print(predictions.shape)
        hickle.dump(predictions, outfile)
    else:
        predictions=predictions[0] # IIRC, there's a useless first dimension in each model output.
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        
        for p in range(predictions.shape[0]):
            misc.imsave(os.path.join(outfile, "%d.png" % p), predictions[p])
    


if __name__=="__main__":
    main(sys.argv)
