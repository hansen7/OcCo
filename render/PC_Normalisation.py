#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, open3d, numpy as np

File_ = open('ModelNet_flist_short.txt', 'w')

if __name__ == "__main__":
    root_dir = "../data/ModelNet_subset/"

    for root, dirs, files in os.walk(root_dir, topdown=False):
        for file in files:
            if '.ply' in file:
                amesh = open3d.io.read_triangle_mesh(os.path.join(root, file))
                out_file_name = os.path.join(root, file).replace('.ply', '_normalised.obj')

                center = amesh.get_center()
                amesh.translate(-center)
                maxR = (np.asarray(amesh.vertices)**2).sum(axis=1).max()**(1/2)
                # we found divided by (2*maxR) has best rendered visualisation results
                amesh.scale(1/(2*maxR))
                open3d.io.write_triangle_mesh(out_file_name, amesh)
                File_.writelines(out_file_name.replace('.obj', '').replace(root_dir, '') + '\n')
                print(out_file_name)
