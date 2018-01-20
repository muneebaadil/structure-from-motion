import numpy as np 
import cv2 

def EstimateFundamentalMatrix(x1,x2):
    if x1.shape[1]==2: #converting to homogenous coordinates if not already
        x1 = cv2.convertPointsToHomogeneous(x1)[:,0,:]
        x2 = cv2.convertPointsToHomogeneous(x2)[:,0,:]

    A = np.zeros((x1.shape[0],9))

    #Constructing A matrix (vectorized)
    x1_ = x1.repeat(3,axis=1)
    x2_ = np.tile(x2, (1,3))

    A = x1_*x2_

    u,s,v = np.linalg.svd(A)
    F = v[-1,:].reshape((3,3),order='F')

    u,s,v = np.linalg.svd(F)
    F = u.dot(np.diag(s).dot(v))

    F = F / np.linalg.norm(F,'fro')
    
    return F 

def NormalizePts(x):
    mus = x[:,:2].mean(axis=0)
    sigma = x[:,:2].std()
    scale = np.sqrt(2.) / sigma

    transMat = np.array([[1,0,mus[0]],[0,1,mus[1]],[0,0,1]])
    scaleMat = np.array([[scale,0,0],[0,scale,0],[0,0,1]])

    T = scaleMat.dot(transMat)

    xNorm = T.dot(x.T).T

    return xNorm, T

def EstimateFundamentalMatrixNormalized(x1,x2): 
    if x1.shape[1]==2: #converting to homogenous coordinates if not already
        x1 = cv2.convertPointsToHomogeneous(x1)[:,0,:]
        x2 = cv2.convertPointsToHomogeneous(x2)[:,0,:]

    x1Norm,T1 = NormalizePts(x1)
    x2Norm,T2 = NormalizePts(x2)

    F = EstimateFundamentalMatrix(x1Norm,x2Norm)

    F = T1.T.dot(F.dot(T2))
    return F

def EstimateFundamentalMatrixRANSAC(img1pts,img2pts,outlierThres,prob=None,iters=None): 
    
    Fs = np.zeros((iters,3,3))
    for i in xrange(iters): 
        mask = np.random.randint(low=0,high=img1pts.shape[0],size=(8,))
        
        img1ptsiter = img1pts[mask]
        img2ptsiter = img2pts[mask]
        Fs[i,:,:] = sfmnp.EstimateFundamentalMatrix(img1ptsiter,img2ptsiter)
        
        #if i%50==0:
        #    print '{}/{} iters done..'.format(i,iters)
        
    return Fs 