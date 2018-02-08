import argparse
import cv2 
import os
from itertools import izip
import pickle as pkl
from utils import * 


def main(opts):
    #choosing appropriate feature extractor.. 
    if opts.featExtractor =='sift': 
        featExtractor = cv2.xfeatures2d.SIFT_create()
    elif opts.featExtractor =='surf': 
        featExtractor = cv2.xfeatures2d.SURF_create()

    #creating output directory if not already existing..
    if not os.path.exists(opts.outDir): 
        os.makedirs(opts.outDir)
    
    #reading image names and paths.. 
    imgNames = [x for x in sorted(os.listdir(opts.dataDir)) if x.split('.')[-1] in opts.ext]
    imgPaths = [os.path.join(opts.dataDir,x) for x in imgNames]

    for i, (imgPath,imgName) in enumerate(izip(imgPaths,imgNames)): 

        #loading images, computing keypoints and descriptors, and saving it..
        img = cv2.imread(imgPath)
        imgName = imgName.split('.')[0] #removing the extension from filename

        kp, desc = featExtractor.detectAndCompute(img,None) 
        kp_ = SerializeKeypoints(kp)
        
        with open(os.path.join(opts.outDir, imgName+'.pkl'), 'wb') as fileobj: 
            pkl.dump((kp_, desc), fileobj)

        if i%opts.printEvery==0: 
            print '{}/{} done..'.format(i+1, len(imgNames))

    return 

def SetArguments(parser):
    parser.add_argument('-dataDir',action='store', type=str, default='../data/fountain-P11/images', dest='dataDir') 
    parser.add_argument('-outDir',action='store', type=str, default='../data/fountain-P11/images/keypoints_descriptors', dest='outDir') 
    parser.add_argument('-printEvery',action='store', type=int, default=1, dest='printEvery') 
    parser.add_argument('-ext',action='store', type=list, default=['jpg','png'], dest='ext') 
    parser.add_argument('-featExtractor',action='store', type=str, default='surf', dest='featExtractor') 
    return 

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    main(opts)