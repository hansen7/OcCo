#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import h5py

def Dataset_Assign(dataset, fname, partial=True, bn=False, few_shot=False):

    def fetch_files(filelist):
        return [item.strip() for item in open(filelist).readlines()]

    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    dataset = dataset.lower()

    if dataset == 'shapenet8':
        NUM_CLASSES = 8
        if partial:
            NUM_TRAINOBJECTS = 231792
            TRAIN_FILES = fetch_files('./data/shapenet/hdf5_partial_1024/train_file.txt')
            VALID_FILES = fetch_files('./data/shapenet/hdf5_partial_1024/valid_file.txt')
        else:
            raise ValueError("For ShapeNet we are only interested in the partial objects recognition")

    elif dataset == 'shapenet10':
        # Number of Objects:  17378
        # Number of Objects:  2492
        NUM_CLASSES, NUM_TRAINOBJECTS = 10, 17378
        TRAIN_FILES = fetch_files('./data/ShapeNet10/Cleaned/train_file.txt')
        VALID_FILES = fetch_files('./data/ShapeNet10/Cleaned/test_file.txt')

    elif dataset == 'modelnet40':
        '''Actually we find that using data from PointNet++: 
        https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
        will increase the accuracy a bit, however to make a fair comparison: we use the same data as 
        the original data provided by PointNet: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'''
        NUM_CLASSES = 40
        if partial:
            NUM_TRAINOBJECTS = 98430
            TRAIN_FILES = fetch_files('./data/modelnet40_pcn/hdf5_partial_1024/train_file.txt')
            VALID_FILES = fetch_files('./data/modelnet40_pcn/hdf5_partial_1024/test_file.txt')
        else:
            VALID_FILES = fetch_files('./data/modelnet40_ply_hdf5_2048/test_files.txt')
            if few_shot:
                TRAIN_FILES = fetch_files('./data/modelnet40_ply_hdf5_2048/few_labels/%s.h5' % fname)
                data, _ = loadh5DataFile('./data/modelnet40_ply_hdf5_2048/few_labels/%s.h5' % fname)
                NUM_TRAINOBJECTS = len(data)
            else:
                NUM_TRAINOBJECTS = 9843
                TRAIN_FILES = fetch_files('./data/modelnet40_ply_hdf5_2048/train_files.txt')

    elif dataset == 'scannet10':
        NUM_CLASSES, NUM_TRAINOBJECTS = 10, 6110
        TRAIN_FILES = fetch_files('./data/ScanNet10/ScanNet_Cleaned/train_file.txt')
        VALID_FILES = fetch_files('./data/ScanNet10/ScanNet_Cleaned/test_file.txt')

    elif dataset == 'scanobjectnn':
        NUM_CLASSES = 15
        if bn:
            TRAIN_FILES = ['./data/ScanNetObjectNN/h5_files/main_split/training_objectdataset' + fname + '.h5']
            VALID_FILES = ['./data/ScanNetObjectNN/h5_files/main_split/test_objectdataset' + fname + '.h5']
            data, _ = loadh5DataFile('./data/ScanNetObjectNN/h5_files/main_split/training_objectdataset' + fname + '.h5')
            NUM_TRAINOBJECTS = len(data)
        else:
            TRAIN_FILES = ['./data/ScanNetObjectNN/h5_files/main_split_nobg/training_objectdataset' + fname + '.h5']
            VALID_FILES = ['./data/ScanNetObjectNN/h5_files/main_split_nobg/test_objectdataset' + fname + '.h5']
            data, _ = loadh5DataFile('./data/ScanNetObjectNN/h5_files/main_split_nobg/training_objectdataset' + fname + '.h5')
            NUM_TRAINOBJECTS = len(data)
    else:
        raise ValueError('dataset not exists')

    return NUM_CLASSES, NUM_TRAINOBJECTS, TRAIN_FILES, VALID_FILES
