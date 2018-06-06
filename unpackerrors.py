#!/usr/bin/env python

# this script quickly dumps all of the elements within an error layer to images.

import hickle
import os
import sys
import numpy as np
from scipy import ndimage
from scipy import misc


def main(args):
    if len(args)!=3:
        print("Usage: unpackerrors.py inputhickle outdir")
        return
    
    # Generally, error hickles are shaped like (1, numframes, numchannels, height, width)
    val=hickle.load(args[1])[0]
    outdir=args[2]
    
    chdirs=[]
    for ch in range(val.shape[1]):
        chdirs.append("%s%d" % (outdir, ch))
        if not os.path.exists(chdirs[ch]):
            os.mkdir(chdirs[ch])
    
    # This should be for every image in the set:
    for i in range(val.shape[0]):
        tmp=val[i][0:3]
        #ttmp=tmp.reshape((tmp.shape[1], tmp.shape[2], tmp.shape[0]), order='F')
        #tttmp=np.zeros((tmp.shape[1], tmp.shape[2], tmp.shape[0]))
        #tttmp[:,:,0]=tmp[0]
        #tttmp[:,:,1]=tmp[1]
        #tttmp[:,:,2]=tmp[2]
        #tmp=val[i][3:6]
        #tttmp1=np.zeros((tmp.shape[1], tmp.shape[2], tmp.shape[0]))
        #tttmp1[:,:,0]=tmp[0]
        #tttmp1[:,:,1]=tmp[1]
        #tttmp1[:,:,2]=tmp[2]
        
        #misc.imsave(os.path.join(chdirs[0], "%d.png" % i), tttmp)
        #misc.imsave(os.path.join(chdirs[1], "%d.png" % i), tttmp1)

        for j in range(val.shape[1]):
            tmp=val[i][j]
            ttmp=tmp.transpose()
            misc.imsave(os.path.join(chdirs[j], "%d.png" % i), tmp)
    print("Done")
            
        
if __name__=="__main__":
    main(sys.argv)
