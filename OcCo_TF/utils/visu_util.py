#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com
#  Original Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d.open3d.io import read_point_cloud
# from open3d.open3d_pybind.io import read_point_cloud


def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for _ in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    filenames = ['airplane.pcd', 'car.pcd', 'chair.pcd', 'lamp.pcd']  # '../demo_data'
    for file in filenames:
        filename = file.replace('.pcd', '')
        pcds = [np.asarray(read_point_cloud('../demo_data/' + file).points)]
        titles = ['viewpoint 1', 'viewpoint 2', 'viewpoint 3']
        plot_pcd_three_views(
            filename, pcds, titles, suptitle=filename, sizes=None, cmap='viridis', zdir='y',
            xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3))
