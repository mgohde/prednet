#!/usr/bin/env python

# genrandset.py -- Randomly selects a subset of model outputs, stimuli, and fixations to work around extreme RAM constraints present in this setup.

import os
import sys
import numpy as np
import shutil


def check_dirs(dirs, printmsg=True):
    for d in dirs:
        if not os.path.exists(d):
            if printmsg:
                print("Path %s does not exist." % d)
            return False
    return True


def get_set(sdir, numelements):
    total=len(os.listdir(sdir))

    # This is probably going to give us a relatively full set;
    rset1=np.random.rand(numelements)*total
    rset2=np.random.rand(numelements)*total
    rset1=np.unique(rset1.astype(int))
    rset2=np.unique(rset2.astype(int))
    return np.union1d(rset1, rset2)[0:numelements]


def filldir(src, dest, s, ext):
    for elem in s:
        shutil.copyfile(os.path.join(src, "%d.%s" % (elem, ext)), os.path.join(dest, "%d.%s" % (elem, ext)))


def getfixations(fixfile, newfixfile, s):
    with open(fixfile, "r") as f:
        lines=f.read().splitlines()
        
        with open(newfixfile, "w") as o:
            for elem in s:
                o.write(lines[elem])
                o.write("\n")

            
def main(args):
    if len(args)!=7:
        print("Usage: genrandset.py stimdir fixationfile salmapdir modeldir numelements outputdir")
        return
    
    stimdir=args[1]
    fixationfile=args[2]
    salmapdir=args[3]
    modeldir=args[4]
    numelements=int(args[5])
    outputdir=args[6]
    modelout=os.path.join(outputdir, "model")
    stimout=os.path.join(outputdir, "stimuli")
    salmapout=os.path.join(outputdir, "goldstandard")
    fixout=os.path.join(outputdir, "fixations.txt")
    
    if not check_dirs([stimdir, fixationfile, modeldir, salmapdir]):
        print("One or more of the directories or files specified does not exist. Exiting...")
        return
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    print("Generating random set...")
    elemset=get_set(stimdir, numelements)
    
    # Now we need to fill directories:
    if not check_dirs([modelout, stimout, salmapout], printmsg=False):
        os.makedirs(modelout)
        os.makedirs(stimout)
        os.makedirs(salmapout)
    
    print("Filling output paths...")
    filldir(modeldir, modelout, elemset, "png") # model files should be PNG images (for now)
    filldir(stimdir, stimout, elemset, "jpg") # stimuli are jpegs extracted from a norpix file.
    filldir(salmapdir, salmapout, elemset, "jpg")
    getfixations(fixationfile, fixout, elemset)
    

if __name__=="__main__":
    main(sys.argv)
