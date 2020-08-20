#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, h5py, json, argparse, numpy as np
from LMDB_DataFlow import lmdb_dataflow
from tqdm import tqdm


def fix2len(point_cloud, fix_length):
    if len(point_cloud) >= fix_length:
        point_cloud = point_cloud[np.random.choice(len(point_cloud), fix_length)]
    else:
        point_cloud = np.concatenate(
            [point_cloud, point_cloud[np.random.choice(len(point_cloud), fix_length - len(point_cloud))]], axis=0)
    return point_cloud


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, default='train')
    parser.add_argument("--lmdb_path", type=str, default=r'../data/modelnet40_pcn/')
    parser.add_argument("--hdf5_path", type=str, default=r'../data/modelnet40_pcn/hdf5_partial_1024')
    parser.add_argument("--partial", action='store_true', help='store partial scan or not')
    parser.add_argument('--num_per_obj', type=int, default=1024)
    parser.add_argument('--num_scan', type=int, default=10)

    args = parser.parse_args()

    lmdb_file = os.path.join(args.lmdb_path, args.f_name + '.lmdb')
    os.system('mkdir -p %s' % args.hdf5_path)
    df_train, num_train = lmdb_dataflow(
        lmdb_path=lmdb_file, batch_size=1, input_size=args.num_per_obj,
        output_size=args.num_per_obj, is_training=False)

    if args.partial:
        print('Now we generate point cloud from partial observed objects.')

    file_per_h5 = 2048 * 4  # of objects within each hdf5 file
    data_gen = df_train.get_data()

    idx = 0
    data_np = np.zeros((file_per_h5, args.num_per_obj, 3))
    label_np = np.zeros((file_per_h5,), dtype=np.int32)
    ids_np = np.chararray((file_per_h5,), itemsize=32)

    # convert label string to integers
    hash_label = json.load(open('../data/shapenet_names.json'))
    f_open = open(os.path.join(args.hdf5_path, '%s_file.txt' % args.f_name), 'a+')

    for i in tqdm(range(num_train)):
        '''each object has eight different views'''

        ids, inputs, npts, gt = next(data_gen)
        object_pc = inputs[0] if args.partial else gt[0]

        if len(object_pc) != args.num_per_obj:
            object_pc = fix2len(object_pc, args.num_per_obj)
        if args.partial:
            data_np[i % file_per_h5, :, :] = object_pc
            label_np[i % file_per_h5] = int(hash_label[(ids[0].split('_')[0])])
            ids_np[i % file_per_h5] = ids[0]  # .split('_')[1]

        else:
            if i % args.num_scan != 0:
                continue
            data_np[(i // args.num_scan) % file_per_h5, :, :] = object_pc
            label_np[(i // args.num_scan) % file_per_h5] = int(hash_label[(ids[0].split('_')[0])])
            ids_np[(i // args.num_scan) % file_per_h5] = ids[0].split('_')[1]

        num_obj_ = i if args.partial else i // args.num_scan

        if num_obj_ - idx * file_per_h5 >= file_per_h5:
            h5_file = os.path.join(args.hdf5_path, '%s%d.h5' % (args.f_name, idx))
            print('the last two objects coordinates, labels and ids:')
            print(data_np[-2:])
            print(label_np[-2:])
            print(ids_np[-2:])
            print('\n')

            hf = h5py.File(h5_file, 'w')
            hf.create_dataset('data', data=data_np)
            hf.create_dataset('label', data=label_np)
            hf.create_dataset('id', data=ids_np)
            hf.close()

            f_open.writelines(h5_file.replace('../', './') + '\n')
            print('%s_%s.h5 has been saved' % (args.f_name, idx))
            print('====================\n\n')
            idx += 1

    '''to deal with the remaining in the end'''
    h5_file = os.path.join(args.hdf5_path, '%s%d.h5' % (args.f_name, idx))
    hf = h5py.File(h5_file, 'w')

    if args.partial:
        label_res = label_np[:num_train % file_per_h5]
        data_res = data_np[:num_train % file_per_h5]
        id_res = ids_np[:num_train % file_per_h5]

    else:
        label_res = label_np[:(num_train // args.num_scan) % file_per_h5]
        data_res = data_np[:(num_train // args.num_scan) % file_per_h5]
        id_res = ids_np[:(num_train // args.num_scan) % file_per_h5]

    print('the remaining  objects coordinates, labels and ids:')
    print(data_res[-2:], '\n', label_res[-2:], '\n', id_res[-2:], '\n\n')

    hf.create_dataset('label', data=label_res)
    hf.create_dataset('data', data=data_res)
    hf.create_dataset('id', data=id_res)
    hf.close()
    print('the last part has been saved into %s_%s.h5' % (args.f_name, idx))

    f_open.writelines(h5_file.replace('../', './'))
    f_open.close()

    print('convert from lmdb to hdf5 has finished')
