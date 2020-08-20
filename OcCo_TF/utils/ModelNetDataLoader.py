import os, torch, h5py, warnings, numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: point cloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {'train': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))],
                     'test': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]}

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class General_CLSDataLoader_HDF5(Dataset):
    def __init__(self, file_list, num_point=1024):
        # self.root = root
        self.num_point = num_point
        self.file_list = file_list
        self.points_list = np.zeros((1, num_point, 3))
        self.labels_list = np.zeros((1,))

        for file in self.file_list:
            # pdb.set_trace()
            # file = os.path.join(root, file)
            # pdb.set_trace()
            data, label = self.loadh5DataFile(file)
            self.points_list = np.concatenate([self.points_list,
                                               data[:, :self.num_point, :]], axis=0)
            self.labels_list = np.concatenate([self.labels_list, label.ravel()], axis=0)

        self.points_list = self.points_list[1:]
        self.labels_list = self.labels_list[1:]
        assert len(self.points_list) == len(self.labels_list)
        print('Number of Objects: ', len(self.labels_list))

    @staticmethod
    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    def __len__(self):
        return len(self.points_list)

    def __getitem__(self, index):

        point_xyz = self.points_list[index][:, 0:3]
        point_label = self.labels_list[index].astype(np.int32)

        return point_xyz, point_label


class ModelNetJigsawDataLoader(Dataset):
    def __init__(self, root=r'./data/modelnet40_ply_hdf5_2048/jigsaw',
                 n_points=1024, split='train', k=3):
        self.npoints = n_points
        self.root = root
        self.split = split
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('train') is not -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('test') is not -1]
        self.points_list = np.zeros((1, n_points, 3))
        self.labels_list = np.zeros((1, n_points))

        for file in self.file_list:
            file = os.path.join(root, file)
            data, label = self.loadh5DataFile(file)
            # data = np.load(root + file)
            self.points_list = np.concatenate([self.points_list, data], axis=0)  # .append(data)
            self.labels_list = np.concatenate([self.labels_list, label], axis=0)
        # self.labels_list.append(label)

        self.points_list = self.points_list[1:]
        self.labels_list = self.labels_list[1:]
        assert len(self.points_list) == len(self.labels_list)
        print('Number of %s Objects: '%self.split, len(self.labels_list))

        # just use the average weights
        self.labelweights = np.ones(k ** 3)

    # pdb.set_trace()

    @staticmethod
    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    def __getitem__(self, index):

        point_set = self.points_list[index][:, 0:3]
        semantic_seg = self.labels_list[index].astype(np.int32)
        # sample_weight = self.labelweights[semantic_seg]

        # return point_set, semantic_seg, sample_weight
        return point_set, semantic_seg

    def __len__(self):
        return len(self.points_list)


if __name__ == '__main__':

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True, )
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
