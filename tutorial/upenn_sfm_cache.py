import numpy as np 
import scipy.io as io 

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import cv2 

def EstimateFundamentalMatrix(x1,x2): 
    A = np.zeros((x1.shape[0],9))
    for i in xrange(x1.shape[0]):
        rowA = np.array([x1[i,0]*x2[i,0], x1[i,0]*x2[i,1], x1[i,0],
              x1[i,1]*x2[i,0], x1[i,1]*x2[i,1], x1[i,1],
              x2[i,0], x2[i,1], 1])
        A[i,:] = rowA

    u,s,v = np.linalg.svd(A)
    F = v[-1,:].reshape((3,3),order='F')

    u,s,v = np.linalg.svd(F)
    F = u.dot(np.diag(s).dot(v))

    F = F / np.linalg.norm(F,'fro')
    
    return F 

def EstimateEssentialMatrix(K,F): 
    E = K.T.dot(F.dot(K))
    u,s,v = np.linalg.svd(E)
    s[0] = 1.
    s[1] = 1.
    s[2] = 0.
    E = u.dot(np.diag(s).dot(v))
    #E = E / np.linalg.norm(E,'fro')
    #probably need to add norm too 
    
    return E

def Vec2Skew(vec): 
    return np.array([[0,-vec[2],vec[1]],[vec[2],0,-vec[0]],[-vec[1],vec[0],0]])

def LinearTriangulate(K, C1, R1, C2, R2, x1, x2): 
    P1 = K.dot(np.hstack((R1,C1)))
    P2 = K.dot(np.hstack((R2,C2)))
    
    X = np.zeros((x1.shape[0],3))
    x1h = np.hstack((x1, np.ones((x1.shape[0],1))))
    x2h = np.hstack((x2, np.ones((x2.shape[0],1))))
    
    for i in xrange(x1h.shape[0]): 
        x1hcross = Vec2Skew(x1h[i,:])
        x2hcross = Vec2Skew(x2h[i,:])

        A = np.vstack((x1hcross.dot(P1),x2hcross.dot(P2)))

        u,s,v = np.linalg.svd(A)

        ans = v[-1,:3] / v[-1,-1]
        X[i,:] = ans 
        
    return X

def LinearPnP(X, x, K): 
    xh = np.hstack((x, np.ones((x.shape[0],1))))
    Xh = np.hstack((X, np.ones((X.shape[0],1))))
    xc = np.linalg.inv(K).dot(xh.T).T

    A = np.zeros((X.shape[0]*3,12))

    for i in xrange(X.shape[0]): 
        A[i*3,:] = np.concatenate((np.zeros((4,)), -Xh[i,:], xc[i,1]*Xh[i,:]))
        A[i*3+1,:] = np.concatenate((Xh[i,:], np.zeros((4,)), -xc[i,0]*Xh[i,:]))
        A[i*3+2,:] = np.concatenate((-xc[i,1]*Xh[i,:], xc[i,0]*Xh[i,:], np.zeros((4,))))    
    
    u,s,v = np.linalg.svd(A)
    P = v[-1,:].reshape((4,3),order='F').T
    R, t = P[:,:3], P[:,-1]
    
    u,s,v = np.linalg.svd(R)
    R = u.dot(v)
    t = t/s[0]

    if np.linalg.det(u.dot(v)) < 0:
        R = R*-1
        t = t*-1

    C = -R.T.dot(t)
    
    return R, t

def ExtractCameraPoses(E):
    assert(np.linalg.matrix_rank(E, tol=1e-7)==2)
    
    u,d,v = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    Rs, Cs = np.zeros((4,3,3)), np.zeros((4,3))
    
    Cs[0] = u[:,-1]
    Rs[0] = u.dot(W.dot(v.T))
    
    Cs[1] = -u[:,-1]
    Rs[1] = u.dot(W.dot(v.T))
    
    Cs[2] = u[:,-1]
    Rs[2] = u.dot(W.T.dot(v.T))
    
    Cs[3] = -u[:,-1]
    Rs[3] = u.dot(W.T.dot(v.T))
    
    for i in xrange(4):        
        if np.linalg.det(Rs[i]) < 0: 
            Cs[i] = -1*Cs[i]
            Rs[i] = -1*Rs[i]
    return Rs, Cs

def Display3DPoints(X): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(X[:,0],X[:,1],X[:,2])    
    return ax

def LoadUPennData(): 
    #LOADING IMAGES, KEYPOINTS ETC..
    var = io.loadmat('variables.mat')
    varPnp = io.loadmat('variablesPnP.mat')

    img1=var['data'][0,0][6]
    img2=var['data'][0,0][7]
    img3=var['data'][0,0][8]

    x1, x2, x3 = var['x1'],var['x2'],var['x3']
    K = var['K']

    C2,R2 = var['C'],var['R']
    C1,R1 = np.ones((3,1)), np.eye(3,3)
    
    return img1,img2,img3, x1,x2,x3, C1,R1, C2,R2,K

def RunSFM(filename=None): 
    #LOADING IMAGES, KEYPOINTS ETC..
    img1,img2,img3,x1,x2,x3,C1,R1,C2,R2=LoadUPennData()
    
    #FUNDAMENTAL MATRIX ESTIMATION
    F = EstimateFundamentalMatrix(x1,x2)
    
    #ESSENTIAL MATRIX ESTIMATION
    E = EstimateEssentialMatrix(K,F)
    
    #POSE ESTIMATION. (CHERIALITY CHECK IS DONE INTERNALLY)
    _, R, t, masktwo = cv2.recoverPose(E, x1, x2, K)
    
    #LINEAR TRIANGULATION
    X = LinearTriangulate(K, np.zeros((3,1)), np.eye(3,3), t, R, x1, x2)
    
    out = {'F':F, 'E':E, 'X':X}
    return out

if __name__=='__main__':
    filename = 'variables.mat'
    RunSFM(filename)
