#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com
#  Generating Training Data of 3D Point Cloud for 3D Jigsaw Puzzles

import os, h5py, numpy as np

'''
The 3D object/block is split into voxels along axes, 
each point is assigned with a voxel label.
'''


def pc_ssl_3djigsaw_gen(pc_xyz, k=2, edge_len=1):
    """
    :param pc_xyz: point cloud, (n_point, 3 + additional feature)
    :param k: number of voxels along each axis
    :param edge_len: length of voxel (cube) edge
    :return: permuted pc, labels
    """
    intervals = [edge_len*2 / k * x - edge_len for x in np.arange(k + 1)]
    assert edge_len >= pc_xyz.__abs__().max()
    indices = np.searchsorted(intervals, pc_xyz, side='left') - 1
    label = indices[:, 0] * k ** 2 + indices[:, 1] * k + indices[:, 2]

    shuffle_indices = np.arange(k ** 3)
    np.random.shuffle(shuffle_indices)
    shuffled_dict = dict()
    for i, d in enumerate(shuffle_indices):
        shuffled_dict[i] = d

    def numberToBase(n, base=k):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(str(int(n % base)))
            n //= base
        return int("".join(digits[::-1]))

    for voxel_id in range(k ** 3):
        selected_points = (label == voxel_id)
        permutated_places = shuffled_dict[voxel_id]
        loc = permutated_places
        center_diff = np.array([(loc // k ** 2) - (voxel_id // k ** 2),
                                (loc // k ** 2) // k - (voxel_id // k ** 2) // k,
                                loc % k - voxel_id % k]) * (2 * edge_len)/k  # + const - edge_len
        pc_xyz[selected_points] = pc_xyz[selected_points] + center_diff

    return pc_xyz, label


if __name__ == "__main__":
    root_dir = r'./data/modelnet40_ply_hdf5_2048'
    dir_path = r'./data/modelnet40_ply_hdf5_2048/jigsaw/k2'
    os.mkdir(dir_path) if not os.path.exists(dir_path) else None

    TRAIN_FILES = [item.strip() for item in open(os.path.join(root_dir, 'train_files.txt')).readlines()]
    VALID_FILES = [item.strip() for item in open(os.path.join(root_dir, 'test_files.txt')).readlines()]


    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]


    def reduce2fix(pc, n_points=1024):
        indices = np.arange(len(pc))
        np.random.shuffle(indices)
        return pc[indices[:n_points]]


    for file_ in VALID_FILES:
        filename = file_.split('/')[-1]
        print(filename)
        data, _ = loadh5DataFile(file_)
        # subsample all point clouds into 1024 points of each
        data = np.apply_along_axis(reduce2fix, axis=1, arr=data)
        shuffled_data = np.zeros_like(data)
        shuffled_label = np.zeros((data.shape[0], data.shape[1]))
        for idx, pc_xyz in enumerate(data):
            pc_xyz, label = pc_ssl_3djigsaw_gen(pc_xyz, k=2, edge_len=1)
            shuffled_data[idx] = pc_xyz
            shuffled_label[idx] = label
        hf = h5py.File(os.path.join(dir_path, filename), 'w')

        hf.create_dataset('label', data=shuffled_label)
        hf.create_dataset('data', data=shuffled_data)
        hf.close()
