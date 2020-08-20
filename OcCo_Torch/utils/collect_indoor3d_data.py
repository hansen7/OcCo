#  Ref: https://github.com/charlesq34/pointnet/blob/master/sem_seg/collect_indoor3d_data.py

import os, sys, indoor3d_util
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(indoor3d_util.DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
# output_folder = os.path.join('../data/stanford_indoor3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
# Ref: https://github.com/charlesq34/pointnet/issues/45
for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy'  # e.g., Area_1_hallway_1.npy
        indoor3d_util.collect_point_label(
            anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')
