import argparse
import cv2 
import os
from itertools import izip
import pickle as pkl
import numpy as np 

def main(opts):
    #choosing appropriate feature extractor.. 
    if opts.featExtractor =='sift': 
        featExtractor = cv2.xfeatures2d.SIFT_create()
    elif opts.featExtractor =='surf': 
        featExtractor = cv2.xfeatures2d.SURF_create()
    
    #reading image names and paths.. 
    imgNames = [x for x in sorted(os.listdir(opts.dataDir)) if x.split('.')[-1] in opts.ext]
    imgPaths = [os.path.join(opts.dataDir,x) for x in imgNames]

    for imgPath,imgName in izip(imgPaths,imgNames): 

        #loading images, computing keypoints and descriptors, and saving it..
        img = cv2.imread(imgPath)
        imgName = imgName.split('.')[0] #removing the extension from filename

        kp1, desc1 = featExtractor.detectAndCompute(img,None) 

        break 
    return 

def SetArguments(parser):
    parser.add_argument('-dataDir',action='store', type=str, default='../data/fountain-P11/images', dest='dataDir') 
    parser.add_argument('-ext',action='store', type=list, default=['jpg','png'], dest='ext') 
    parser.add_argument('-featExtractor',action='store', type=str, default='surf', dest='featExtractor') 
    return 

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    main(opts)