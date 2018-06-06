#!/usr/bin/env python3

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
