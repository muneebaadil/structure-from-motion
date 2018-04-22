import numpy as np 
import cv2 
import argparse
import pickle
import os 

from utils import * 
import pdb 

class SFM(object): 
    def __init__(self, opts): 
        self.opts = opts
        self.point_cloud = None

        self.images_dir = os.path.join(opts.data_dir,opts.dataset, 'images')
        self.feat_dir = os.path.join(opts.data_dir, opts.dataset, 'features', opts.features)
        self.img_names = [x.split('.')[0] for x in sorted(os.listdir(self.images_dir))]

        self.image_data = []
        self.matcher = getattr(cv2, opts.matcher)(crossCheck=opts.cross_check)

        if opts.calibration_mat == 'benchmark': 
            self.K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
        else: 
            raise NotImplementedError
        
    def _LoadFeatures(self, name): 
        with open(os.path.join(self.feat_dir,'kp_{}.pkl'.format(name)),'r') as f: 
            kp = pickle.load(f)
        kp = DeserializeKeypoints(kp)

        with open(os.path.join(self.feat_dir,'desc_{}.pkl'.format(name)),'r') as f: 
            desc = pickle.load(f)

        return kp, desc 

    def _GetAlignedMatches(self,kp1,desc1,kp2,desc2,matches):
        img1idx = np.array([m.queryIdx for m in matches])
        img2idx = np.array([m.trainIdx for m in matches])

        #filtering out the keypoints that were NOT matched. 
        kp1_ = (np.array(kp1))[img1idx]
        kp2_ = (np.array(kp2))[img2idx]

        #retreiving the image coordinates of matched keypoints
        img1pts = np.array([kp.pt for kp in kp1_])
        img2pts = np.array([kp.pt for kp in kp2_])

        return img1pts,img2pts

    def _BaselinePoseEstimation(self, name1, name2): 

        kp1, desc1 = self._LoadFeatures(name1)
        kp2, desc2 = self._LoadFeatures(name2)  

        matches = self.matcher.match(desc1,desc2)
        matches = sorted(matches, key = lambda x:x.distance)

        img1pts, img2pts = self._GetAlignedMatches(kp1,desc1,kp2,desc2,matches)
        
        F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,
                                        param1=opts.outlier_thres,param2=opts.fund_prob)
        mask = mask.astype(bool).flatten()

        E = self.K.T.dot(F.dot(self.K))
        _,R,t,_ = cv2.recoverPose(E,img1pts[mask],img2pts[mask],self.K)

        return R,t

    def _AddViewsToReconstruction(self): 
        pass 

    def _ToPly(self): 
        pass

    def Run(self):
        name1, name2 = self.img_names[0], self.img_names[1]

        R,t = self._BaselinePoseEstimation(name1, name2)

        while False: 
            self._NewViewPoseEstimation()

        #self.ToPly()
        

def SetArguments(parser): 

    #directory stuff
    parser.add_argument('--data_dir',action='store',type=str,default='../data/',dest='data_dir') 
    parser.add_argument('--dataset',action='store',type=str,default='fountain-P11',dest='dataset') 
    parser.add_argument('--features',action='store',type=str,default='SURF',dest='features') 
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher') 
    parser.add_argument('--cross_check',action='store',type=bool,default=True,dest='cross_check') 
    parser.add_argument('--out_dir',action='store',type=str,default='../results/',dest='out_dir') 

    #computing parameters
    parser.add_argument('--calibration_mat',action='store',type=str,default='benchmark',dest='calibration_mat')
    parser.add_argument('--fund_method',action='store',type=str,default='FM_RANSAC',dest='fund_method')
    parser.add_argument('--outlier_thres',action='store',type=float,default=.9,dest='outlier_thres')
    parser.add_argument('--fund_prob',action='store',type=float,default=.9,dest='fund_prob')

    #misc
    parser.add_argument('--plot_error',action='store',type=bool,default=False,dest='plot_error')  
    parser.add_argument('--verbose',action='store',type=bool,default=True,dest='verbose')  

def PostprocessArgs(opts): 
    opts.fund_method = getattr(cv2,opts.fund_method)

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)
    
    sfm = SFM(opts)
    sfm.Run()