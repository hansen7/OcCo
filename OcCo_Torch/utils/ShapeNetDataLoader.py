#  Copyright (c) 2020. Author: Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ShapeNetDataLoader.py
import os, json, torch, warnings, numpy as np
from PC_Augmentation import pc_normalize
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


class PartNormalDataset(Dataset):
    """
    Data Source: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
    """
    def __init__(self, root, num_point=2048, split='train', use_normal=False):
        self.catfile = os.path.join(root, 'synsetoffset2category.txt')
        self.use_normal = use_normal
        self.num_point = num_point
        self.cache_size = 20000
        self.datapath = []
        self.root = root
        self.cache = {}
        self.meta = {}
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # self.cat -> {'class name': syn_id, ...}
        # self.meta -> {'class name': file list, ...}
        # self.classes -> {'class name': class id, ...}
        # self.datapath -> [('class name', single file) , ...]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        train_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'))
        test_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'))
        val_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'))
        
        for item in self.cat:
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            self.meta[item] = []

            if split is 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split is 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s [Option: ]. Exiting...' % split)
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35],
                            'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29],
                            'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Lamp': [24, 25, 26, 27],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Knife': [22, 23],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 
                            'Chair': [12, 13, 14, 15]}

    @staticmethod
    def read_fns(path):
        with open(path, 'r') as file:
            ids = set([str(d.split('/')[2]) for d in json.load(file)])
        return ids

    def __getitem__(self, index):
        if index in self.cache:
            pts, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat, pt = fn[0], np.loadtxt(fn[1]).astype(np.float32)
            cls = np.array([self.classes[cat]]).astype(np.int32)
            pts = pt[:, :6] if self.use_normal else pt[:, :3]
            seg = pt[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pts, cls, seg)

        choice = np.random.choice(len(seg), self.num_point, replace=True)
        pts[:, 0:3] = pc_normalize(pts[:, 0:3])
        pts, seg = pts[choice, :], seg[choice]

        return pts, cls, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == "__main__":

    root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    TRAIN_DATASET = PartNormalDataset(root=root, num_point=2048, split='trainval', use_normal=False)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=24, shuffle=True, num_workers=4)

    for i, data in enumerate(trainDataLoader):
        points, label, target = data

