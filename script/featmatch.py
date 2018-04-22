import cv2 
import numpy as np 
import pickle 
import argparse
import os 

from utils import * 

def FeatMatch(opts): 
    img_names = sorted(os.listdir(opts.data_dir))
    img_paths = [os.path.join(opts.data_dir, x) for x in img_names if \
                x.split('.')[-1] in opts.ext]
    
    feat_out_dir = os.path.join(opts.out_dir,'features',opts.features)
    if not os.path.exists(feat_out_dir): 
        os.makedirs(feat_out_dir)

    for i, img_path in enumerate(img_paths): 
        img = cv2.imread(img_path)
        img_name = img_names[i].split('.')[0]
        img = img[:,:,::-1]

        feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
        kp, desc = feat.detectAndCompute(img,None)

        kp_ = SerializeKeypoints(kp)
        
        with open(os.path.join(feat_out_dir, 'kp_{}.pkl'.format(img_name)),'wb') as out:
            pickle.dump(kp_, out)

        with open(os.path.join(feat_out_dir, 'desc_{}.pkl'.format(img_name)),'wb') as out:
            pickle.dump(desc, out)

        if opts.save_results: 
            raise NotImplementedError

        if (i % opts.print_every) == 0:
            print '{}/{} features done..'.format(i+1,len(img_paths))

def SetArguments(parser): 

    #directories stuff
    parser.add_argument('--data_dir',action='store',type=str,default='../data/fountain-P11/images/',
                        dest='data_dir') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext') 
    parser.add_argument('--out_dir',action='store',type=str,default='../data/fountain-P11/',
                        dest='out_dir') 

    #feature matching args
    parser.add_argument('--features',action='store', type=str, default='SURF', dest='features') 
    
    #misc
    parser.add_argument('--print_every',action='store', type=int, default=1, dest='print_every')
    parser.add_argument('--save_results',action='store', type=str, default=False, dest='save_results')  

def PostprocessArgs(opts): 
    opts.ext = [x for x in opts.ext.split(',')]

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    FeatMatch(opts)