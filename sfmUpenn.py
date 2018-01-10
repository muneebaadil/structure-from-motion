import numpy as np 
import scipy.io as io 

def EstimateFundamentalMatrix(x1,x2): 
    A = np.zeros((8,9))
    for i in xrange(8):
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

def linearTriangulate(K, C1, R1, C2, R2, x1, x2): 
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

def RunSFM(filename): 
    variables = io.loadmat(filename)
    pass 

if __name__=='__main__':
    filename = 'variables.mat'
    RunSFM(filename)