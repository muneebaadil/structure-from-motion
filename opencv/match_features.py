import cv2, numpy as np
import argparse, os, pickle as pkl
from utils import *
import pdb

def main(opts): 
    #if output path does not exist, creating it..
    if not os.path.exists(opts.outDir): 
        os.makedirs(opts.outDir)
    
    #reading input keypoint&descriptors filenames..
    KDnames = sorted(os.listdir(opts.dataDir))
    KDpaths = [os.path.join(opts.dataDir, x) for x in KDnames]
    
    #setting up matcher's setup
    matcher = cv2.BFMatcher(crossCheck=opts.crossCheck)
    out = {} 

    #matching all possible combinations of descriptors (ignoring the order)
    numDone = 0 
    for i in xrange(len(KDpaths)): 
        for j in xrange(i+1, len(KDpaths)): 
            
            #Reading two keypoints and descriptors to be matched..
            with open(KDpaths[i], 'rb') as fileobj:
                t = pkl.load(fileobj)
                kp1, desc1 = t
                kp1 = DeserializeKeypoints(kp1)

            with open(KDpaths[j], 'rb') as fileobj: 
                t = pkl.load(fileobj)
                kp2, desc2 = t 
                kp2 = DeserializeKeypoints(kp2)

            #brute force descriptor matching.. 
            matches = matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x:x.distance)

            out[(KDnames[i], KDnames[j])] = matches

            #Optional verbosity..
            numDone += 1
            if numDone%opts.printEvery==0: 
                n = len(KDpaths)
                print '{}/{} done..'.format(numDone, (n*(n-1))/2)
    
    #Serealizing matches dictionary so it can be saved using pickle..
    out_ = SerializeMatchesDict(out)
    
    with open(os.path.join(opts.outDir, 'matches.pkl'), 'wb') as fileobj: 
        pkl.dump(out_, fileobj)
    return 

def SetArguments(parser): 
    parser.add_argument('-dataDir',action='store',type=str,default='../data/fountain-P11/images/keypoints_descriptors',dest='dataDir') 
    parser.add_argument('-outDir',action='store',type=str,default='../data/fountain-P11/images/matches',dest='outDir') 
    parser.add_argument('-printEvery',action='store', type=int, default=1, dest='printEvery') 
    parser.add_argument('-crossCheck',action='store', type=bool, default=True, dest='crossCheck') 
    
if __name__ =='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    main(opts)