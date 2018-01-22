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

    F = F / F[-1,-1]
    
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
    if img1pts.shape[1]==2: 
        #converting to homogenous coordinates if not already
        img1pts = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
        img2pts = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

    bestInliers, bestF, bestmask = 0, None, None

    for i in xrange(iters): 
        
        #Selecting 8 random points
        mask = np.random.randint(low=0,high=img1pts.shape[0],size=(8,))
        img1ptsiter = img1pts[mask]
        img2ptsiter = img2pts[mask]

        #Fitting fundamental matrix and evaluating error 
        Fiter = EstimateFundamentalMatrix(img1ptsiter,img2ptsiter)
        err = SampsonError(Fiter,img1pts,img2pts)
        mask = err < outlierThres
        numInliers = np.sum(mask)

        #Updating best measurements if appropriate 
        if bestInliers < numInliers: 
            bestInliers = numInliers
            bestF = Fiter
            bestmask = mask

    #Final least squares fit on all inliers found.. 
    F = EstimateFundamentalMatrix(img1pts[bestmask], img2pts[bestmask])    
    
    return F, bestmask

def ComputeEpiline(pts, index, F): 
    """
    pts: (n,3) points matrix
    
    lines: (n,3) lines matrix"""
    
    if pts.shape[1]==2: 
        #converting to homogenous coordinates if not already
        pts = cv2.convertPointsToHomogeneous(pts)[:,0,:]
        
    if index==1: 
        lines = F.dot(pts.T) 
    elif index==2: 
        lines = F.T.dot(pts.T)
    
    return lines.T

def SampsonError(F,x1,x2): 
    num = np.sum(x1.dot(F) * x2,axis=-1)

    F_src = np.dot(F, x1.T)
    Ft_dst = np.dot(F.T, x2.T)

    dst_F_src = np.sum(x2 * F_src.T, axis=1)
    
    return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + 
    F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)

def ExtractCameraPoses(E): 
    u,d,v = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    Rs, Cs = np.zeros((4,3,3)), np.zeros((4,3))
    
    t = u[:,-1]
    R1 = u.dot(W.dot(v))
    R2 = u.dot(W.T.dot(v))

    if np.linalg.det(R1) < 0: 
        R1 = R1 * -1

    if np.linalg.det(R2) < 0: 
        R2 = R2 * -1
    
    return R1,R2,t

def TransformCoordPts(X,R,t): 
    return (R.dot(X.T)+t).T

def CountFrontOfBothCameras(X, R, t): 
    isfrontcam1 = X[:,-1] > 0
    isfrontcam2 = TransformCoordPts(X,R,t)[:,-1] > 0
    
    return np.sum(isfrontcam1 & isfrontcam2)

def DisambiguateCameraPose(configSet):  
    maxfrontpts = -1 
    
    for R,t,pts3d in configSet: 
        count = sfmnp.CountFrontOfBothCameras(pts3d,R,t)
        
        if count > maxfrontpts: 
            maxfrontpts = count
            bestR,bestt = R,t
    
    return bestR,bestt,maxfrontpts

def TriangulatePts(img1pts,img2pts,P1,P2): 
    if img1pts.shape[1]==2: 
        #converting to homogenous coordinates if not already
        img1pts = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
        img2pts = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]    
    
    A = []
    out = np.zeros((img1pts.shape[0],4))
    
    for i,(img1pt, img2pt) in enumerate(izip(img1pts,img2pts)): 
        img1pt_cross, img2pt_cross = Vec2Skew(img1pt), Vec2Skew(img2pt)
        
        A.append(img1pt_cross.dot(P1))
        A.append(img2pt_cross.dot(P2))
        A = np.concatenate(A,axis=0)
        
        u,s,v = np.linalg.svd(A)
        out[i,:] = v[-1,:]
    return out 