#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import h5py, numpy as np
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from open3d.open3d.io import read_point_cloud, write_point_cloud


def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.array(pcd.points)


def save_pcd(filename, points):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    write_point_cloud(filename, pcd)


def shuffle_data(data, labels):
    """ Shuffle data and labels """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def loadh5DataFile(PathtoFile):
    f = h5py.File(PathtoFile, 'r')
    return f['data'][:], f['label'][:]


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        name='data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        name='label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()
