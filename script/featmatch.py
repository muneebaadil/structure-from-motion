import cv2 
import numpy as np 
import pickle 
import argparse
import os 
from time import time

from utils import * 

def FeatMatch(opts, data_files=[]): 
    
    if len(data_files) == 0: 
        img_names = sorted(os.listdir(opts.data_dir))
        img_paths = [os.path.join(opts.data_dir, x) for x in img_names if \
                    x.split('.')[-1] in opts.ext]
    
    else: 
        img_paths = data_files
        img_names = sorted([x.split('/')[-1] for x in data_files])
        
    feat_out_dir = os.path.join(opts.out_dir,'features',opts.features)
    matches_out_dir = os.path.join(opts.out_dir,'matches',opts.matcher)

    if not os.path.exists(feat_out_dir): 
        os.makedirs(feat_out_dir)
    if not os.path.exists(matches_out_dir): 
        os.makedirs(matches_out_dir)
    
    data = []
    t1 = time()
    for i, img_path in enumerate(img_paths): 
        img = cv2.imread(img_path)
        img_name = img_names[i].split('.')[0]
        img = img[:,:,::-1]

        feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
        kp, desc = feat.detectAndCompute(img,None)
        data.append((img_name, kp, desc))

        kp_ = SerializeKeypoints(kp)
        
        with open(os.path.join(feat_out_dir, 'kp_{}.pkl'.format(img_name)),'wb') as out:
            pickle.dump(kp_, out)

        with open(os.path.join(feat_out_dir, 'desc_{}.pkl'.format(img_name)),'wb') as out:
            pickle.dump(desc, out)

        if opts.save_results: 
            raise NotImplementedError

        t2 = time()

        if (i % opts.print_every) == 0:    
            print 'FEATURES DONE: {0}/{1} [time={2:.2f}s]'.format(i+1,len(img_paths),t2-t1)

        t1 = time()

    num_done = 0 
    num_matches = ((len(img_paths)-1) * (len(img_paths))) / 2

    t1 = time()
    for i in xrange(len(data)): 
        for j in xrange(i+1, len(data)): 
            img_name1, kp1, desc1 = data[i]
            img_name2, kp2, desc2 = data[j]

            matcher = getattr(cv2,opts.matcher)(crossCheck=opts.cross_check)
            matches = matcher.match(desc1,desc2)

            matches = sorted(matches, key = lambda x:x.distance)
            matches_ = SerializeMatches(matches)

            pickle_path = os.path.join(matches_out_dir, 'match_{}_{}.pkl'.format(img_name1,
                                                                                 img_name2))
            with open(pickle_path,'wb') as out:
                pickle.dump(matches_, out)

            num_done += 1 
            t2 = time()

            if (num_done % opts.print_every) == 0: 
                print 'MATCHES DONE: {0}/{1} [time={2:.2f}s]'.format(num_done, num_matches, t2-t1)

            t1 = time()
            


def SetArguments(parser): 

    #directories stuff
    parser.add_argument('--data_files',action='store',type=str,default='',dest='data_files') 
    parser.add_argument('--data_dir',action='store',type=str,default='../data/fountain-P11/images/',
                        dest='data_dir',help='directory containing images (default: ../data/\
                        fountain-P11/images/)') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext',
                        help='comma seperated string of allowed image extensions \
                        (default: jpg,png)') 
    parser.add_argument('--out_dir',action='store',type=str,default='../data/fountain-P11/',
                        dest='out_dir',help='root directory to store results in \
                        (default: ../data/fountain-P11)') 

    #feature matching args
    parser.add_argument('--features',action='store', type=str, default='SURF', dest='features',
                        help='[SIFT|SURF] Feature algorithm to use (default: SURF)') 
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use \
                        (default: BFMatcher)') 
    parser.add_argument('--cross_check',action='store',type=bool,default=True,dest='cross_check',
                        help='[True|False] Whether to cross check feature matching or not \
                        (default: True)') 
    
    #misc
    parser.add_argument('--print_every',action='store', type=int, default=1, dest='print_every',
                        help='[1,+inf] print progress every print_every seconds, -1 to disable \
                        (default: 1)')
    parser.add_argument('--save_results',action='store', type=str, default=False, 
                        dest='save_results',help='[True|False] whether to save images with\
                        keypoints drawn on them (default: False)')  

def PostprocessArgs(opts): 
    opts.ext = [x for x in opts.ext.split(',')]
    
    opts.data_files_ = []
    if opts.data_files != '': 
        opts.data_files_ = opts.data_files.split(',')
    opts.data_files = opts.data_files_

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)

    FeatMatch(opts, opts.data_files)