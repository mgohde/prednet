#!/usr/bin/env python

import nnetmodel
import os
import sys
import hickle
import numpy as np

from scipy.misc import imsave


def main(args):
    if len(args)!=5:
        print("Usage: runnnetmodel [predict/train] <args>")
        print("If command is predict, then args may be the following:")
        print("inputhickle prior outputdir")
        print("If command is train, then args may be the following:")
        print("inputhickle refhickle weightsfilename")
        return
    
    command=args[1]
        
    if command=="train":
        inputhickle=args[2]
        refhickle=args[3]
        weightsfilename=args[4]
        
        print("Loading data...")
        inv=hickle.load(inputhickle)
        inv=inv[:, 0:400, :, :, :]
        print(inv.shape)
        ref=hickle.load(refhickle)
        ref=ref[0:400]
        print(ref.shape)
        
        model=nnetmodel.nnetmodel(inv.shape, ref.shape)
        model.train(inv, ref, 5)
        print("Dumping model weights to disk...")
        model.saveweights(weightsfilename)
        
    
    elif command=="predict":
        inputhickle=args[2]
        weights=args[3]
        outputdir=args[4]
        
        # We just ignore the prior for now.
        print("Loading data...")
        inv=hickle.load(inputhickle)
        inv=inv[:, 201:, :, :, :]
        model=nnetmodel.nnetmodel(inv.shape, (inv.shape[1], 80*60))
        model.loadweights(weights)
        prediction=model.predict(inv)
        
        print(prediction.shape)
        outpred=np.zeros((prediction.shape[0], 60, 80)).astype(np.float16)
        print(prediction.shape)
        print("Dumping predictions...")
        for i in range(prediction.shape[0]):
            #outpred[i]=np.reshape(prediction[i], (80, 60))
            imsave(os.path.join(outputdir, "%d.png" % i), np.reshape(prediction[i], (60, 80)))
        
        
    
    else:
        pass


if __name__=="__main__":
    main(sys.argv)
