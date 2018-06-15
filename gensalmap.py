#!/usr/bin/env python

import os
import sys
import numpy as np
import scipy
from scipy.misc import imsave
from scipy.ndimage import filters


imsize=(480, 640)


def ptscale(pt):
    # Assumption: the specific data passed was from my earlier collection runs and thus needs to be readjusted relative to a ... 2048x1152 viewing surface? This seems roughly right.
    topleftx=2048/2-640/2
    toplefty=1152/2-480/2
    newpt=(pt[0]-topleftx, pt[1]-toplefty)
    return newpt


def main(args):
    if len(args)!=3:
        print("Usage: gensalmap infile outdir")
        return
    
    infile=args[1]
    outdir=args[2]
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    print("working")
    with open(infile, "r") as f:
        lines=f.read().splitlines()
        for l in lines:
            arr=eval(l)
            outarr=np.zeros(imsize)
            
            for elem in arr[1:]:
                pt=ptscale(elem)
                #print(pt)
                newpt=(int(round(pt[0])), int(round(pt[1])))
                if newpt[1]<imsize[0] and newpt[1]>0 and newpt[0]<imsize[1] and newpt[0]>0:
                    outarr[newpt[1]][newpt[0]]=1.0
            
            newoutarr=filters.gaussian_filter(outarr, 32)
            imsave(os.path.join(outdir, "%d.png" % arr[0]), newoutarr)
    print("done")


if __name__=="__main__":
    main(sys.argv)
