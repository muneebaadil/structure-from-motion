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

def PlotCamera(R,t,ax,scale=.5,depth=.5):
    C = -t #camera center (in world coordinate system)
    
    #plotting x-axis of camera coordinate system (red line)
    ax.plot3D(xs=[C[0],R[0,0]+C[0]],ys=[C[1],R[1,0]+C[1]],zs=[C[2],R[2,0]+C[2]],c='r')
    #plotting y-axis of camera coordinate system (green line)
    ax.plot3D(xs=[C[0],R[0,1]+C[0]],ys=[C[1],R[1,1]+C[1]],zs=[C[2],R[2,1]+C[2]],c='g')
    #plotting z-axis of camera coordinate system (blue line)
    ax.plot3D(xs=[C[0],R[0,2]+C[0]],ys=[C[1],R[1,2]+C[1]],zs=[C[2],R[2,2]+C[2]],c='b')
    
    #generating 5 corners of camera polygon 
    pt1 = np.array([[0,0,0]]).T #camera centre
    pt2 = np.array([[scale,-scale,depth]]).T #upper right 
    pt3 = np.array([[scale,scale,depth]]).T #lower right 
    pt4 = np.array([[-scale,-scale,depth]]).T #upper left
    pt5 = np.array([[-scale,scale,depth]]).T #lower left
    pts = np.concatenate((pt1,pt2,pt3,pt4,pt5),axis=-1)
    
    #Transforming to world-coordinate system
    pts = R.dot(pts)+C[:,np.newaxis]
    ax.scatter3D(xs=pts[0,:],ys=pts[1,:],zs=pts[2,:],c='k')
    
    #Generating a list of vertices to be connected in polygon
    verts = [[pts[:,0],pts[:,1],pts[:,2]], [pts[:,0],pts[:,2],pts[:,-1]],
            [pts[:,0],pts[:,-1],pts[:,-2]],[pts[:,0],pts[:,-2],pts[:,1]]]
    
    #Generating a polygon now..
    ax.add_collection3d(Poly3DCollection(verts, facecolors='grey',
                                         linewidths=1, edgecolors='k', alpha=.25))

def DrawCorrespondences(img, ptsTrue, ptsReproj, ax, drawOnly=50): 
    ax.imshow(img)
    
    randidx = np.random.choice(ptsTrue.shape[0],size=(drawOnly,),replace=False)
    ptsTrue_, ptsReproj_ = ptsTrue[randidx], ptsReproj[randidx]
    
    colors = colors=np.random.rand(drawOnly,3)
    
    ax.scatter(ptsTrue_[:,0],ptsTrue_[:,1],marker='x',c=colors)
    ax.scatter(ptsReproj_[:,0],ptsReproj_[:,1],marker='.',c=colors)