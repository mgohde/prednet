#!/usr/bin/env python
'''
Code to process the caltech pedestrian detection dataset.
'''

import os
import sys
import requests
from bs4 import BeautifulSoup
import urllib
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
from kitti_settings import *
import random


# The caltech dataset is entirely 640x480, so we need to use a resolution compatible with that:
#desired_im_sz = (480, 640)
desired_im_sz=(120, 160)
# For this example, we're going to train on set02 and test on set03.
categories = ['set02', 'set03']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = [('city', '2011_09_26_drive_0005_sync')]
test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

# Download raw zip files by scraping KITTI website
def download_data():
    base_dir = os.path.join(DATA_DIR, 'raw/')
    if not os.path.exists(base_dir): os.mkdir(base_dir)
    for c in categories:
        url = "http://www.cvlibs.net/datasets/kitti/raw_data.php?type=" + c
        r = requests.get(url)
        soup = BeautifulSoup(r.content)
        drive_list = soup.find_all("h3")
        drive_list = [d.text[:d.text.find(' ')] for d in drive_list]
        print "Downloading set: " + c
        c_dir = base_dir + c + '/'
        if not os.path.exists(c_dir): os.mkdir(c_dir)
        for i, d in enumerate(drive_list):
            print str(i+1) + '/' + str(len(drive_list)) + ": " + d
            url = "http://kitti.is.tue.mpg.de/kitti/raw_data/" + d + "/" + d + "_sync.zip"
            urllib.urlretrieve(url, filename=c_dir + d + "_sync.zip")


# unzip images
def extract_data():
    for c in categories:
        c_dir = os.path.join(DATA_DIR, 'raw/', c + '/')
        _, _, zip_files = os.walk(c_dir).next()
        for f in zip_files:
            print 'unpacking: ' + f
            spec_folder = f[:10] + '/' + f[:-4] + '/image_03/data*'
            command = 'unzip -qq ' + c_dir + f + ' ' + spec_folder + ' -d ' + c_dir + f[:-4]
            os.system(command)


# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        _, folders, _ = os.walk(c_dir).next()
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw/', category, folder, folder[:10], folder, 'image_03/data/')
            _, _, files = os.walk(im_dir).next()
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files)

        print 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images'
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))


# This is useful because it returns image filenames in order wtihout needing to do a sort.
def imfilenames(imdir):
    n=len(os.listdir(imdir))
    return ["%s.jpg" % s for s in range(n)]


def process_caltech(set02dir, set03dir):
    print("Getting file lists...")
    set02=os.listdir(set02dir)
    set03=os.listdir(set03dir)
    
    testtrain={l: [] for l in ["train", "test", "val"]}
    
    print("Setting up training and testing sets...")
    for v in set02:
        if random.randint(0, 1):
            testtrain['train'].extend([(os.path.join(set02dir, v), s) for s in imfilenames(os.path.join(set02dir, v))])
        else:
            testtrain["test"].extend([(os.path.join(set02dir, v), s) for s in imfilenames(os.path.join(set02dir, v))])
    
    print("Setting up validation sets...")
    for v in set03:
        if random.randint(0, 1):
            testtrain["val"].extend([(os.path.join(set03dir, v), s) for s in imfilenames(os.path.join(set03dir, v))])
    
    
    # Now that we have all of the training/testing divisions et up, we need to make some data hickles:
    for elem in testtrain:
        div=testtrain[elem]
        ims=[]
        sources=[]
        
        for src, im in div:
            ims.append(os.path.join(src, im))
            sources.append(src)
        
        print("Loading image array for split/division %s" % div)
        # Create an image array:
        X=np.zeros((len(ims),)+desired_im_sz+(3,), np.uint8)
        for i, imfile in enumerate(ims):
            im=imread(imfile)
            X[i]=process_im(im, desired_im_sz)
        
        hkl.dump(X, "X_%s.hkl" % elem)
        hkl.dump(sources, "sources_%s.hkl" % elem)
            
        

# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage: process_caltech.py path_to_train_dir path_to_val_dir")
    else:
        process_caltech(sys.argv[1], sys.argv[2])
