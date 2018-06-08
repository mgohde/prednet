#!/usr/bin/env python

# score.py -- A deepgaze model and testing fixture that should be able to compare against groundtruth data collected
# during the summer of 2017.

# This effectively wraps an image read and some scaling.

from __future__ import print_function
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# Now import the good stuff:
import pysaliency
import pysaliency.external_datasets

# Pysaliency presents stimuli as FileStimuli objects. These appear to wrap a set of files in addition to some various
# accounting data. A FileStimuli object features a list of filenames (named "filenames"), in addition to the shapes, sizes, indices, counts, etc. of all of the objects present. 

# Fixations are presented as FixationTrains objects. Each FixatoinTrains (aaauuugh) has a set of x, x_int and y, y_int arrays representing coordinates. There are also x_hist and y_hist histograms of these coordinate values. 

# It appears that model outputs should be in the form

# Based on the above, we can read all appropriate data by

# This scoring system works in acordance to some of the principles established in Saliency Benchmarking: Separating Models, Maps and Metrics. 

def ptscale(pt):
    # Assumption: the specific data passed was from my earlier collection runs and thus needs to be readjusted relative to a ... 2048x1152 viewing surface? This seems roughly right.
    topleftx=2048/2-640/2
    toplefty=1152/2-480/2
    newpt=(pt[0]-topleftx, pt[1]-toplefty)
    return newpt


def getFixations(fixfile):
    f=open(fixfile, 'r')
    lines=f.read().splitlines()
    f.close()
    
    # FixationTrains objects consist of lists of lists of fixations. For example:
    # x=[[1, 2, 3], [4, 5, 6]]
    # y=[[2, 2, 2], [3, 3, 3]]
    # n=[0, 1]
    # t=[[0], [0]]
    xlist=[]
    ylist=[]
    nlist=[]
    tlist=[]
    subjectlist=[]
    
    for i, l in enumerate(lines):
        toks=eval(l)
        toks=toks[1:] # Omit the frame number.
        
        curxlist=[]
        curylist=[]
        #curnlist=[i]
        nlist.append(i)
        tl=[] # Let's just say that this took place over no time.
        
        for i, t in enumerate(toks):
            if len(t)!=0:
                if type(t[0]) is tuple:
                    print("tuple time?")
                newt=ptscale(t)
                if i%2 and not (newt[0]>=640 or newt[0]<=0 or newt[1]>=480 or newt[1]<=0):
                    curxlist.append(newt[0])
                    curylist.append(newt[1])
                    #curnlist.append(int(i))
                    tl.append(0)
        
        xlist.append(np.array(curxlist))
        ylist.append(np.array(curylist))
        tlist.append(np.array(tl))
        #nlist.append(np.array(curnlist, dtype=int))
        subjectlist.append(np.array([0]))
    
    # Now that we've reorganized, the data, let's reorganize the data yet again:
    fixations = pysaliency.FixationTrains.from_fixation_trains(xlist, ylist, tlist, nlist, subjectlist)
    return fixations
            
        
# Rather conveniently, we can just import our model from a directory of (images?)
def main(args):
    if len(args)!=5:
        print("Usage: score.py stimdir modeldir fixationfile goldstandarddir")
        return
    
    stimdir=args[1]
    modeldir=args[2]
    fixfile=args[3]
    goldstandarddir=args[4]
    # This expects a directory of pngs, jpgs, tiff, mats, or npys.
    # The stimulus list is given as a set of files.
    # More specifically, that set is given as a FileStimuli object
    stims=os.listdir(stimdir)
    # Assumption: we're dealing with an extracted norpix dataset or some such.
    sortedstims=[os.path.join(stimdir, s) for s in stims]
    stimuli=pysaliency.FileStimuli(sortedstims)
    #Further note: the number of model images must be equal to the number of stimuli. They must also se the same base naming convention (less file extensions) as said stimuli.
    print("Loading data into model...")
    model=pysaliency.SaliencyMapModelFromDirectory(stimuli, modeldir)
    
    print("Loading gold standard model...")
    goldstandard=pysaliency.SaliencyMapModelFromDirectory(stimuli, goldstandarddir)
    
    print("Reading fixations...")
    fixations=getFixations(fixfile)
    
    # To evaluate the model, we need to call one of the evaluation functions in the model itself.
    print("score:")
    print("AUC:")
    print(model.AUC(stimuli, fixations, nonfixations="uniform")) # nonfixations may also be "shuffled"
    print("Fixation based KL divergence:")
    print(model.fixation_based_KL_divergence(stimuli, fixations, nonfixations="uniform"))
    print("Image based KL divergence:")
    # TODO: consider setting the model parameter to false, since the passed gold standard *should* potentially be probabalistic.
    print(model.image_based_kl_divergence(stimuli, goldstandard))
    print("NSS:")
    print(model.NSS(stimuli, fixations))

if __name__=="__main__":
    main(sys.argv)
