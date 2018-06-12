#!/usr/bin/env python

# this script exists to pack a directory of saliency maps into a hickle usable by the neural net model.
# ASSUMPTION: these maps have already been downscaled to 80x60. Thanks, memory constraints!

import os
import sys
import numpy as np
from scipy.misc import imread
import hickle


def main(args):
    if len(args)!=3:
        print("Usage: packsaliency.py inputdir outputhickle")
        return
    
    indir=args[1]
    outname=args[2]
    
    inputlist=np.zeros((len(os.listdir(indir)), 80*60)).astype(np.float16)
    for i, f in enumerate(os.listdir(indir)):
        im=imread(os.path.join(indir, "%d.jpg" % i)).astype(np.float16)
        im=im/255.0 # normalize?
        inputlist[i]=im.flatten()
    hickle.dump(inputlist, outname)


if __name__=="__main__":
    main(sys.argv)
