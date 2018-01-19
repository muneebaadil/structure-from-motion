import numpy as np 

def EstimateFundamentalMatrix(x1,x2):
    A = np.zeros((x1.shape[0],9))
    for i in xrange(x1.shape[0]):
        rowA = np.array([x1[i,0]*x2[i,0], x1[i,0]*x2[i,1], x1[i,0],
              x1[i,1]*x2[i,0], x1[i,1]*x2[i,1], x1[i,1],
              x2[i,0], x2[i,1], 1])
        A[i,:] = rowA

    u,s,v = np.linalg.svd(A)
    F = v[-1,:].reshape((3,3),order='F')

    # u,s,v = np.linalg.svd(F)
    # F = u.dot(np.diag(s).dot(v))

    # F = F / np.linalg.norm(F,'fro')
    
    return F 