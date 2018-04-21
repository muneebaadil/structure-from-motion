import cv2
import numpy as np 
from utils import * 

def GetFundamentalMatrix(img1pts,img2pts,outlierThres=.1,prob=.99): 
    F,mask=cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC,param1=outlierThres,param2=prob)
    mask=mask.astype(bool).flatten()

    return F,mask


def ExtractCameraPoses(K,F,img1pts,img2pts):
    E = K.T.dot(F.dot(K))
    _,R,t,mask2=cv2.recoverPose(E,img1pts,img2pts,K)

    return R,t,mask2

def RunSFM():
    #loading images here 
    img1 = cv2.imread('./data/fountain-P11/images/0005.jpg')
    img2 = cv2.imread('./data/fountain-P11/images/0006.jpg')

    #1/4. FEATURE MATCHING
    img1pts,img2pts,matches=GetImageMatches(img1,img2) 

    #2/4. FUNDAMENTAL MATRIX ESTIMATION
    F,mask=GetFundamentalMatrix(img1pts,img2pts)
    
    #3/4. CAMERA POSE ESTIMATION
    K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
    R,t,mask2=ExtractCameraPoses(K,F,img1pts[mask],img2pts[mask])

    #4/4. TRIANGULATION. 
    pts3d=GetTriangulatedPts(img1pts[mask],img2pts[mask],K,R,t)

    #Finally, saving 3d points in .ply format to view in meshlab software
    pts2ply(pts3d)

if __name__=='__main__': 
    RunSFM()