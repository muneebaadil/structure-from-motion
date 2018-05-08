#FOUNTAIN P11 DATASET
#SURF
python featmatch.py --data_dir ../data/fountain-P11/images/ --out_dir ../data/fountain-P11/ --features SURF
#SIFT
python featmatch.py --data_dir ../data/fountain-P11/images/ --out_dir ../data/fountain-P11/ --features SIFT

#ENTRY P10 DATASET
#SURF
python featmatch.py --data_dir ../data/entry-P10/images/ --out_dir ../data/entry-P10/ --features SURF
#SIFT
python featmatch.py --data_dir ../data/fountain-P11/images/ --out_dir ../data/entry-P10/ --features SIFT

#HERZ JESUS P8 DATASET
#SURF
python featmatch.py --data_dir ../data/Herz-Jesus-P8/images/ --out_dir ../data/Herz-Jesus-P8/ --features SURF
#SIFT
python featmatch.py --data_dir ../data/fountain-P11/images/ --out_dir ../data/Herz-Jesus-P8/ --features SIFT