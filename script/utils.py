import numpy as np 
import cv2 
import pdb
from itertools import izip 

def SerializeKeypoints(kp): 
    """Serialize list of keypoint objects so it can be saved using pickle
    
    Args: 
    kp: List of keypoint objects 
    
    Returns: 
    out: Serialized list of keypoint objects"""

    out = []
    for kp_ in kp: 
        temp = (kp_.pt, kp_.size, kp_.angle, kp_.response, kp_.octave, kp_.class_id)
        out.append(temp)

    return out

def DeserializeKeypoints(kp): 
    """Deserialize list of keypoint objects so it can be converted back to
    native opencv's format.
    
    Args: 
    kp: List of serialized keypoint objects 
    
    Returns: 
    out: Deserialized list of keypoint objects"""

    out = []
    for point in kp:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2],
         _response=point[3], _octave=point[4], _class_id=point[5]) 
        out.append(temp)

    return out

def SerializeMatches(matches): 
    """Serializes dictionary of matches so it can be saved using pickle
    
    Args: 
    matches: List of matches object
    
    Returns: 
    out: Serialized list of matches object"""

    out = []
    for match in matches: 
        matchTemp = (match.queryIdx, match.trainIdx, match.imgIdx, match.distance) 
        out.append(matchTemp)
    return out

def DeserializeMatches(matches): 
    """Deserialize dictionary of matches so it can be converted back to 
    native opencv's format. 
    
    Args: 
    matches: Serialized list of matches object
    
    Returns: 
    out: List of matches object"""

    out = []
    for match in matches:
        out.append(cv2.DMatch(match[0],match[1],match[2],match[3])) 
    return out

def GetAlignedMatches(kp1,desc1,kp2,desc2,matches):
    """Aligns the keypoints so that a row of first keypoints corresponds to the same row 
    of another keypoints
    
    Args: 
    kp1: List of keypoints from first (left) image
    desc1: List of desciptros from first (left) image
    kp2: List of keypoints from second (right) image
    desc2: List of desciptros from second (right) image
    matches: List of matches object
    
    Returns: 
    img1pts, img2pts: (n,2) array where img1pts[i] corresponds to img2pts[i] 
    """

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

    return img1pts,img2pts

def pts2ply(pts,colors,filename='out.ply'): 
    """Saves an ndarray of 3D coordinates (in meshlab format)"""

    with open(filename,'w') as f: 
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
        
        #pdb.set_trace()
        colors = colors.astype(int)
        for pt, cl in izip(pts,colors): 
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                cl[0],cl[1],cl[2]))

def DrawCorrespondences(img, ptsTrue, ptsReproj, ax, drawOnly=50): 
    """
    Draws correspondence between ground truth and reprojected feature point

    Args: 
    ptsTrue, ptsReproj: (n,2) numpy array
    ax: matplotlib axis object
    drawOnly: max number of random points to draw

    Returns: 
    ax: matplotlib axis object
    """
    ax.imshow(img)
    
    randidx = np.random.choice(ptsTrue.shape[0],size=(drawOnly,),replace=False)
    ptsTrue_, ptsReproj_ = ptsTrue[randidx], ptsReproj[randidx]
    
    colors = colors=np.random.rand(drawOnly,3)
    
    ax.scatter(ptsTrue_[:,0],ptsTrue_[:,1],marker='x',c='r',linewidths=.1, label='Ground Truths')
    ax.scatter(ptsReproj_[:,0],ptsReproj_[:,1],marker='x',c='b',linewidths=.1, label='Reprojected')
    ax.legend()

    return ax