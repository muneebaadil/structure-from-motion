import cv2 
import numpy as np 
import argparse
from utils import *
import pickle as pkl
import pdb

def main(opts): 
    #Loading 5th and 6th image data only (hardcoded for now)..
    with open('../data/fountain-P11/images/keypoints_descriptors/0005.pkl') as fileobj: 
        data1 = pkl.load(fileobj)
        kp1, desc1 = data1
        kp1 = DeserializeKeypoints(kp1)
    
    with open('../data/fountain-P11/images/keypoints_descriptors/0006.pkl') as fileobj: 
        data2 = pkl.load(fileobj)
        kp2, desc2 = data2
        kp2 = DeserializeKeypoints(kp2)

    with open('../data/fountain-P11/images/matches/matches.pkl') as fileobj: 
        matches = pkl.load(fileobj)
        matches = matches[('0005.pkl','0006.pkl')]
        matches = DeserializeMatchesDict(matches)
        
    #2/4. FUNDAMENTAL MATRIX ESTIMATION
    img1pts,img2pts = GetAlignedMatches(kp1,desc1,kp2,desc2,matches)
    F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC,param1=opts.outlierThres,
                                    param2=opts.fundProb)
    
    #3/4. CAMERA POSE ESTIMATION
    K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]]) #hardcoded for now, have to generalize.. 
    E = K.T.dot(F.dot(K))
    retval,R,t,mask2 = cv2.recoverPose(E,img1pts[mask],img2pts[mask],K)

    #4/4. TRIANGULATION. 
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts[mask])[:,0,:]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts[mask])[:,0,:]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

    pts4d = cv2.triangulatePoints(np.eye(3,4),np.hstack((R,t)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

    #Finally, saving 3d points in .ply format to view in meshlab software
    pts2ply(pts3d)


def SetArguments(parser): 
    parser.add_argument('-dataDir',action='store',type=str,default='../data/fountain-P11/images/keypoints_descriptors',dest='dataDir') 
    parser.add_argument('-outName',action='store',type=str,default='../data/fountain-P11/images/matches',dest='outDir') 
    parser.add_argument('-printEvery',action='store', type=int, default=1, dest='printEvery') 
    parser.add_argument('-crossCheck',action='store', type=bool, default=True, dest='crossCheck') 

    parser.add_argument('-outlierThres',action='store', type=float, default=.1, dest='outlierThres') 
    parser.add_argument('-fundProb',action='store', type=float, default=.99, dest='fundProb') 
    return 

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    main(opts)