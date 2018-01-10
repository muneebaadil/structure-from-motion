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

def RunSFM(filename): 
    variables = io.loadmat(filename)
    pass 

if __name__=='__main__':
    filename = 'variables.mat'
    RunSFM(filename)