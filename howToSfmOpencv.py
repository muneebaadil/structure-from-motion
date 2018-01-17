import cv2
import numpy as np 
from utils import * 

#loading images here 
img1 = cv2.imread('./data/fountain-P11/images/0005.jpg')
img2 = cv2.imread('./data/fountain-P11/images/0006.jpg')

#1/3. FEATURE MATCHING
surfer=cv2.xfeatures2d.SIFT_create()
kp1, desc1 = surfer.detectAndCompute(img1,None)
kp2, desc2 = surfer.detectAndCompute(img2,None)

matcher = cv2.BFMatcher(crossCheck=True)
matches = matcher.match(desc1, desc2)

matches = sorted(matches, key = lambda x:x.distance)

#retrieving corresponding indices of keypoints (in both images) from matches.  
img1idx = np.array([m.queryIdx for m in matches])
img2idx = np.array([m.trainIdx for m in matches])

#filtering out the keypoints that were NOT matched. 
kp1_ = (np.array(kp1))[img1idx]
kp2_ = (np.array(kp2))[img2idx]

#retreiving the image coordinates of matched keypoints
img1pts = np.array([kp.pt for kp in kp1_])
img2pts = np.array([kp.pt for kp in kp2_])

#2/3. CAMERA MATRICES ESTIMATION
F,mask=cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC,param1=.1,param2=.99)
mask=mask.astype(bool).flatten()

K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
E = K.T.dot(F.dot(K))

_,R,t,mask2=cv2.recoverPose(E,img1pts[mask],img2pts[mask],K)
#np.linalg.det(R)

#3/3. TRIANGULATION. 
img1ptsHom = cv2.convertPointsToHomogeneous(img1pts[mask])[:,0,:]
img2ptsHom = cv2.convertPointsToHomogeneous(img2pts[mask])[:,0,:]

img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

pts4d = cv2.triangulatePoints(np.eye(3,4),np.hstack((R,t)),img1ptsNorm.T,img2ptsNorm.T)
pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

pts2ply(pts3d)