#!/usr/bin/env python
# This script generates a scaled down prior distribution based on a set of fixation maps.

from scipy import misc
import numpy as np
import os
import sys
import hickle
from scipy.misc import logsumexp


def main(args):
    if len(args)!=3:
        print("Usage: priorgen.py inputdir outputfile")
        return
    
    indir=args[1]
    outfile=args[2]
    
    innames=[os.path.join(indir, f) for f in os.listdir(indir)]
    
    baseim=misc.imread(innames[0])
    z=np.zeros(baseim.shape)
    
    print("Summing images...")
    for i in innames:
        z=z+misc.imread(innames[0])
    
    print("Computing average...")
    z=z/np.max(z)
    
    print("Dumping result to disk...")
    hickle.dump(z, outfile)
    
if __name__=="__main__":
    main(sys.argv)
