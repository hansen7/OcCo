#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com
#  Modify the path w.r.t your own settings


def Dataset_Loc(dataset, fname, partial=True, bn=False, few_shot=False):
    def fetch_files(filelist):
        return [item.strip() for item in open(filelist).readlines()]

    dataset = dataset.lower()

    if dataset == 'shapenet8':
        NUM_CLASSES = 8
        if partial:
            TRAIN_FILES = fetch_files('./data/shapenet/hdf5_partial_1024/train_file.txt')
            VALID_FILES = fetch_files('./data/shapenet/hdf5_partial_1024/valid_file.txt')
        else:
            raise ValueError("For ShapeNet we are only interested in the partial objects recognition")

    elif dataset == 'shapenet10':
        NUM_CLASSES = 10
        TRAIN_FILES = fetch_files('./data/ShapeNet10/Cleaned/train_file.txt')
        VALID_FILES = fetch_files('./data/ShapeNet10/Cleaned/test_file.txt')

    # elif dataset == 'modelnet10':
    # 	NUM_CLASSES = 10
    # 	TRAIN_FILES = fetch_files('./data/ModelNet10/Cleaned/train_file.txt')
    # 	VALID_FILES = fetch_files('./data/ModelNet10/Cleaned/test_file.txt')

    elif dataset == 'modelnet40':
        '''Actually we find that using data from PointNet++: '''
        NUM_CLASSES = 40
        if partial:
            TRAIN_FILES = fetch_files('./data/modelnet40_pcn/hdf5_partial_1024/train_file.txt')
            VALID_FILES = fetch_files('./data/modelnet40_pcn/hdf5_partial_1024/test_file.txt')
        else:
            VALID_FILES = fetch_files('./data/modelnet40_ply_hdf5_2048/test_files.txt')
            if few_shot:
                TRAIN_FILES = fetch_files('./data/modelnet40_ply_hdf5_2048/few_labels/%s.h5' % fname)
            else:
                TRAIN_FILES = fetch_files('./data/modelnet40_ply_hdf5_2048/train_files.txt')

    elif dataset == 'scannet10':
        NUM_CLASSES = 10
        TRAIN_FILES = fetch_files('./data/ScanNet10/ScanNet_Cleaned/train_file.txt')
        VALID_FILES = fetch_files('./data/ScanNet10/ScanNet_Cleaned/test_file.txt')

    elif dataset == 'scanobjectnn':
        NUM_CLASSES = 15
        if bn:
            TRAIN_FILES = ['./data/ScanNetObjectNN/h5_files/main_split/training_objectdataset' + fname + '_1024.h5']
            VALID_FILES = ['./data/ScanNetObjectNN/h5_files/main_split/test_objectdataset' + fname + '_1024.h5']

        else:
            TRAIN_FILES = ['./data/ScanNetObjectNN/h5_files/main_split_nobg/training_objectdataset' + fname + '_1024.h5']
            VALID_FILES = ['./data/ScanNetObjectNN/h5_files/main_split_nobg/test_objectdataset' + fname + '_1024.h5']

    else:
        raise ValueError('dataset not exists')

    return NUM_CLASSES, TRAIN_FILES, VALID_FILES
