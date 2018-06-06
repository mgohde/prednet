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
from kitti_settings import *


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
        predictions=predictions[0] # IIRC, there's a useless first dimension in each model output.
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        
        for p in range(predictions.shape[0]):
            misc.imsave(predictions[p], os.path.join(outfile, "%d.png" % p))
        return
    
    indata=hickle.load(infile)
    m=model.PredSaliencyModel(None, prior) # Prior is passed as a filename and the first argument has yet to be defined.
    
    # Todo: figure out if "indata" is actually formatted correctly for this application.
    predictions=m.predict(indata)
    
    if outopt=='-o':
        hickle.dump(predictions, outfile)
    else:
        predictions=predictions[0] # IIRC, there's a useless first dimension in each model output.
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        
        for p in range(predictions.shape[0]):
            misc.imsave(predictions[p], os.path.join(outfile, "%d.png" % p))
    


if __name__=="__main__":
    main(sys.argv)
