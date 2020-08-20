#  Copyright (c) 2020. Author: Hanchen Wang, hw501@cam.ac.uk

import os, argparse, numpy as np
from tensorpack import DataFlow, dataflow
from open3d.open3d.io import read_triangle_mesh, read_point_cloud


def sample_from_mesh(filename, num_samples=16384):
    pcd = read_triangle_mesh(filename).sample_points_uniformly(number_of_points=num_samples)
    return np.array(pcd.points)


class pcd_df(DataFlow):
    def __init__(self, model_list, num_scans, partial_dir, complete_dir, num_partial_points=1024):
        self.model_list = [_file for _file in model_list if 'train' in _file]
        self.num_scans = num_scans
        self.partial_dir = partial_dir
        self.complete_dir = complete_dir
        self.num_ppoints = num_partial_points

    def size(self):
        return len(self.model_list) * self.num_scans

    @staticmethod
    def read_pcd(filename):
        pcd = read_point_cloud(filename)
        return np.array(pcd.points)

    def get_data(self):
        for model_id in self.model_list:
            complete = sample_from_mesh(os.path.join(self.complete_dir, '%s.obj' % model_id))
            for i in range(self.num_scans):
                partial = self.read_pcd(os.path.join(self.partial_dir, model_id + '_%d.pcd' % i))
                partial = partial[np.random.choice(len(partial), self.num_ppoints)]
                yield model_id.replace('/', '_'), partial, complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path', default=r'../render/ModelNet_flist_normalised.txt')
    parser.add_argument('--num_scans', type=int, default=10)
    parser.add_argument('--partial_dir', default=r'../render/dump_modelnet_normalised_supercoarse/pcd')
    parser.add_argument('--complete_dir', default=r'../data/ModelNet40')
    parser.add_argument('--output_file', default=r'../data/ModelNet40_train_1024_supercoarse.lmdb')
    args = parser.parse_args()

    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    df = pcd_df(model_list, args.num_scans, args.partial_dir, args.complete_dir)
    if os.path.exists(args.output_file):
        os.system('rm %s' % args.output_file)
    dataflow.LMDBSerializer.save(df, args.output_file)
