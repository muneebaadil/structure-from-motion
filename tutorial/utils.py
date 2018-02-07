import cv2 
import numpy as np 
from itertools import izip 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def pts2ply(pts,filename='out.ply'): 
    f = open(filename,'w')
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(pts.shape[0]))
    
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    
    f.write('end_header\n')
    
    for pt in pts: 
        f.write('{} {} {} 255 255 255\n'.format(pt[0],pt[1],pt[2]))
    f.close()

def drawlines(img1,img2,lines,pts1,pts2,drawOnly=None,linesize=3,circlesize=10):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:-1]

    img1_, img2_ = np.copy(img1), np.copy(img2)

    drawOnly = lines.shape[0] if (drawOnly is None) else drawOnly

    i = 0 
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        
        img1_ = cv2.line(img1_, (x0,y0), (x1,y1), color,linesize)
        img1_ = cv2.circle(img1_,tuple(pt1.astype(int)),circlesize,color,-1)
        img2_ = cv2.circle(img2_,tuple(pt2.astype(int)),circlesize,color,-1)

        i += 1 
        
        if i > drawOnly: 
            break 

    return img1_,img2_

def GetEpipair(F, pts1, pts2,drawOnly=None):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2,drawOnly)
    epipair = np.concatenate((img5,img6),axis=1)
    
    return epipair

def GetFundamentalMatrix(img1,img2):
    surfer=cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = surfer.detectAndCompute(img1.copy(),None)
    kp2, desc2 = surfer.detectAndCompute(img2.copy(),None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key = lambda x:x.distance)

    img1pts = np.array([kp1[m.queryIdx].pt for m in matches])
    img2pts = np.array([kp2[m.trainIdx].pt for m in matches])

    F,mask = cv2.findFundamentalMat(img1pts, img2pts, method=cv2.FM_RANSAC,
                               param1=.1, param2=.99)
    return F, mask, img1pts, img2pts

def DrawMatchesCustom(img1,img2,kp1,kp2,F,drawOnly=None): 
    fig,ax=plt.subplots(ncols=2,figsize=(11,4))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    
    drawOnly = kp1.shape[0] if (drawOnly is None) else drawOnly
    i = 0 
    for pt1,pt2 in izip(kp1,kp2):
        color = tuple(list(np.random.uniform(size=(3,))))
        #print color 
        ax[0].plot(pt1[0],pt1[1],'x',c=color)
        ax[1].plot(pt2[0],pt2[1],'x',c=color)
        
        line=cv2.computeCorrespondEpilines(pt2[np.newaxis],2,F)
        r=line.reshape((-1,))
        
        lefty = -r[2]/r[1]
        righty = (-r[2]-(r[0]*img2.shape[1]))/r[1]
        
        ax[0].plot([0,img2.shape[0]], [lefty,righty],c=color)
        
        i += 1
        if i > drawOnly: 
            break 

def PlotCamera(R,t,ax,scale=.5,depth=.5,faceColor='grey'):
    C = -t #camera center (in world coordinate system)

    #Generating camera coordinate axes
    axes = np.zeros((3,6))
    axes[0,1], axes[1,3],axes[2,5] = 1,1,1
    
    #Transforming to world coordinate system 
    axes = R.T.dot(axes)+C[:,np.newaxis]

    #Plotting axes
    ax.plot3D(xs=axes[0,:2],ys=axes[1,:2],zs=axes[2,:2],c='r')
    ax.plot3D(xs=axes[0,2:4],ys=axes[1,2:4],zs=axes[2,2:4],c='g')
    ax.plot3D(xs=axes[0,4:],ys=axes[1,4:],zs=axes[2,4:],c='b')

    #generating 5 corners of camera polygon 
    pt1 = np.array([[0,0,0]]).T #camera centre
    pt2 = np.array([[scale,-scale,depth]]).T #upper right 
    pt3 = np.array([[scale,scale,depth]]).T #lower right 
    pt4 = np.array([[-scale,-scale,depth]]).T #upper left
    pt5 = np.array([[-scale,scale,depth]]).T #lower left
    pts = np.concatenate((pt1,pt2,pt3,pt4,pt5),axis=-1)
    
    #Transforming to world-coordinate system
    pts = R.T.dot(pts)+C[:,np.newaxis]
    ax.scatter3D(xs=pts[0,:],ys=pts[1,:],zs=pts[2,:],c='k')
    
    #Generating a list of vertices to be connected in polygon
    verts = [[pts[:,0],pts[:,1],pts[:,2]], [pts[:,0],pts[:,2],pts[:,-1]],
            [pts[:,0],pts[:,-1],pts[:,-2]],[pts[:,0],pts[:,-2],pts[:,1]]]
    
    #Generating a polygon now..
    ax.add_collection3d(Poly3DCollection(verts, facecolors=faceColor,
                                         linewidths=1, edgecolors='k', alpha=.25))

def DrawCorrespondences(img, ptsTrue, ptsReproj, ax, drawOnly=50): 
    ax.imshow(img)
    
    randidx = np.random.choice(ptsTrue.shape[0],size=(drawOnly,),replace=False)
    ptsTrue_, ptsReproj_ = ptsTrue[randidx], ptsReproj[randidx]
    
    colors = colors=np.random.rand(drawOnly,3)
    
    ax.scatter(ptsTrue_[:,0],ptsTrue_[:,1],marker='x',c=colors)
    ax.scatter(ptsReproj_[:,0],ptsReproj_[:,1],marker='.',c=colors)

def GetImageMatches(img1,img2):
    surfer=cv2.xfeatures2d.SURF_create()
    kp1, desc1 = surfer.detectAndCompute(img1,None)
    kp2, desc2 = surfer.detectAndCompute(img2,None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key = lambda x:x.distance)

    return kp1,desc1,kp2,desc2,matches

def GetAlignedMatches(kp1,desc1,kp2,desc2,matches):

    #Sorting in case matches array isn't already sorted
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

    return img1pts,img2pts,img1idx,img2idx

def Find2D3DMatches(desc1,img1idx,desc2,img2idx,desc3,kp3,mask,pts3d):
    #Picking only those descriptors for which 3D point is available
    desc1_3D = desc1[img1idx][mask]
    desc2_3D = desc2[img2idx][mask]

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc3, np.concatenate((desc1_3D,desc2_3D),axis=0))

    #Filtering out matched 2D keypoints from new view 
    img3idx = np.array([m.queryIdx for m in matches])
    kp3_ = (np.array(kp3))[img3idx]
    img3pts = np.array([kp.pt for kp in kp3_])

    #Filtering out matched 3D already triangulated points 
    pts3didx = np.array([m.trainIdx for m in matches])
    pts3didx[pts3didx >= pts3d.shape[0]] = pts3didx[pts3didx >= pts3d.shape[0]] - pts3d.shape[0]
    pts3d_ = pts3d[pts3didx]

    return img3pts, pts3d_

def GetTriangulatedPts(img1pts,img2pts,K,R,t): 
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

    pts4d = cv2.triangulatePoints(np.eye(3,4),np.hstack((R,t)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

    return pts3d