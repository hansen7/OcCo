#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import numpy as np
import os, open3d, sys

LOG_F = open(r'./scale_sum_modelnet40raw.txt', 'w+')
open3d.utility.set_verbosity_level = 0

def log_string(msg):
    print(msg)
    LOG_F.writelines(msg + '\n')


if __name__ == "__main__":

    lmdb_f = r'./data/shapenet/train.lmdb'
    modelnet_raw_path = r'./data/modelnet40_raw/'
    shapenet_raw_path = r'./data/ShapeNet_raw/'
    modelnet40_pn_processed_f = r'./data/'

    off_set, max_radius = 0, 0

    '''=== ModelNet40 ==='''
    log_string('=== ModelNet40 Raw ===\n\n\n')
    for root, dirs, files in os.walk(modelnet_raw_path):
        for name in files:
            if '.ply' in name:
                mesh = open3d.io.read_triangle_mesh(os.path.join(root, name))
                off_set_bias = (mesh.get_center()**2).sum()

                if off_set_bias > off_set:
                    off_set = off_set_bias
                    log_string('update offset: %f by %s' % (off_set, os.path.join(root, name)))
                    radius_bias = (np.asarray(mesh.vertices)**2).sum(axis=1).max()

                    if radius_bias > max_radius:
                        max_radius = radius_bias
                        log_string('update max radius: %f by %s' %(max_radius, os.path.join(root, name)))
    log_string('\n\n\n=== sum for ShapeNetCorev2 ===')
    log_string('===offset:%f,  radius:%f===\n\n\n'%(off_set, max_radius))

    sys.exit('finish computing ModelNet40')


    '''=== ShapeNetCore ==='''
    log_string('=== now on ShapeNetCorev2 ===\n\n\n')
    for root, dirs, files in os.walk(shapenet_raw_path):
        for name in files:
            if '.obj' in name:
                mesh = open3d.io.read_triangle_mesh(os.path.join(root, name))
                off_set_bias = (mesh.get_center()**2).sum()
                if off_set_bias > off_set:
                    off_set = off_set_bias
                    log_string('update offset: %f by %s' % (off_set, os.path.join(root, name)))

                radius_bias = (np.asarray(mesh.vertices)**2).sum(axis=1).max()

                if radius_bias > max_radius:
                    max_radius = radius_bias
                    log_string('update max radius: %f by %s' %(max_radius, os.path.join(root, name)))

    log_string('\n\n\n=== sum for ShapeNetCorev2 ===')
    log_string('===offset:%f,  radius:%f===\n\n\n'%(off_set, max_radius))

    sys.exit('finish computing ShapeNetCorev2')

    '''=== PCN ==='''
    log_string('===now on PCN cleaned subset of ShapeNet===\n\n\n')
    df_train, num_train = lmdb_dataflow(lmdb_path = lmdb_f, batch_size=1,
                                        input_size=3000, output_size=16384, is_training=True)
    train_gen = df_train.get_data()

    for idx in range(231792):
        ids, _, _, gt = next(train_gen)
        off_set_bias = (gt.mean(axis=1)**2).sum()

        if off_set_bias > off_set:
            off_set = off_set_bias
            log_string('update offset: %f by %d, %s' % (off_set, idx, ids))

        radius_bias = (gt**2).sum(axis=2).max()

        if radius_bias > max_radius:
            max_radius = radius_bias
            log_string('update max radius: %f by %d, %s' %(max_radius, idx, ids))

    log_string('\n\n\n===for PCN cleaned subset of ShapeNet===')
    log_string('===offset:%f,  radius:%f===\n\n\n'%(off_set, max_radius))

