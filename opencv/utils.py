import numpy as np 
import cv2 

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

def SerializeMatchesDict(matches): 
    """Serializes dictionary of matches so it can be saved using pickle
    
    Args: 
    matches: Dictionary of (pair of image names, list of matches object) key value pairs
    
    Returns: 
    out: Serialized dictionary"""

    out = {}
    for k,v in matches.items(): 
        temp = []
        for match in v: 
            matchTemp = (match.queryIdx, match.trainIdx, match.imgIdx, match.distance) 
            temp.append(matchTemp)
        out[k] = temp
    return out

def DeserializeMatchesDict(matches): 
    """Deserialize dictionary of matches so it can be converted back to 
    native opencv's format. 
    
    Args: 
    matches: Serialized dictionary (same format as output of SerealizeMatches)
    
    Returns: 
    out: Deserealized dictionary (same format as input of SerealizeMatches method)"""

    out = {}
    for k,v in matches.items(): 
        temp = []
        for match in v:
            temp.append(cv2.DMatch(match[0],match[1],match[2],match[3]))
        out[k] = temp 
    return out