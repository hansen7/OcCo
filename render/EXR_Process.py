#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/wentaoyuan/pcn/blob/master/render/process_exr.py

import os, array, Imath, OpenEXR, argparse, numpy as np, matplotlib.pyplot as plt
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector
from open3d.open3d.io import write_point_cloud
from tqdm import tqdm


def read_exr(exr_path, height, width):
    """from EXR files to extract depth information"""
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    return depth


def depth2pcd(depth, intrinsics, pose):
    """backproject to points cloud from 2.5D depth images"""
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)  # upside-down

    y, x = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default=r'./ModelNet_Flist.txt')
    parser.add_argument('--intrinsics', default=r'./intrinsics.txt')
    parser.add_argument('--output_dir', default=r'./dump')
    parser.add_argument('--num_scans', default=10)
    args = parser.parse_args()

    with open(args.list_file) as file:
        model_list = file.read().splitlines()
    intrinsics = np.loadtxt(args.intrinsics_file)
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)

    counter = 0

    for model_id in tqdm(model_list):
        depth_dir = os.path.join(args.output_dir, 'depth')
        pcd_dir = os.path.join(args.output_dir, 'pcd', model_id)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(pcd_dir, exist_ok=True)
        for i in range(args.num_scans):
            counter += 1

            exr_path = os.path.join(args.output_dir, 'exr', model_id + '_%d.exr' % i)
            pose_path = os.path.join(args.output_dir, 'pose', model_id + '_%d.txt' % i)
            depth_path = os.path.join(args.output_dir, 'depth', model_id + '_%d.npy' % i)
            depth = read_exr(exr_path, height, width)
            np.save(depth_path, np.array(depth))
            # depth_img = Image(np.uint16(depth * 1000))
            # write_image(os.path.join(depth_dir, '%s_%d.png' % (model_id,  i)), depth_img)

            if counter % 1 == 0:
                counter = 1
                plt.figure(figsize=(16, 10))
                plt.imshow(np.array(depth), cmap='inferno')
                plt.colorbar(label='Normalised Distance to Camera')
                plt.title('Depth image')
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.tight_layout()
                plt.savefig(os.path.join(depth_dir, model_id.split('/')[-1] + '_%d.png' % i), dpi=200)
                plt.close()

            pose = np.loadtxt(pose_path)
            points = depth2pcd(depth, intrinsics, pose)
            try:
                normalised_points = points/((points**2).sum(axis=1).max())
                pcd = PointCloud()

            except:
                print('there is an exception in the partial normalised process: ', model_id, i)

            # if there is something wrong with the normalisation process, it will automatically save
            # the previous normalised point cloud for the current objects
            # pcd.points = Vector3dVector(normalised_points)

            pcd.points = Vector3dVector(points)
            write_point_cloud(pcd_dir + '_%d.pcd' % i, pcd)

        # os.removedirs(depth_dir)
        os.removedirs(pcd_dir)
