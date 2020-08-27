#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/wentaoyuan/pcn/blob/master/models/pcn_cd.py
#  Ref: https://github.com/AnTao97/UnsupervisedPointCloudReconstruction/blob/master/model.py
#  Sanity Check: https://github.com/vinits5/learning3d/blob/master/models/pcn.py

import sys, torch, itertools, numpy as np, torch.nn as nn
from pcn_util import PCNEncoder
sys.path.append("../chamfer_distance")
from chamfer_distance import ChamferDistance


class get_model(nn.Module):
    def __init__(self, **kwargs):
        super(get_model, self).__init__()

        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__dict__.update(kwargs)  # to update args, num_coarse, grid_size, grid_scale

        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]

        self.feat = PCNEncoder(global_feat=True, channel=3)

        # batch normalisation will destroy limit the expression
        self.folding1 = nn.Sequential(
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(1024+2+3, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1))

    def build_grid(self, batch_size):
        # a simpler alternative would be: torch.meshgrid()
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        # substitute for tf.tile:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/tile
        # Ref: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:  # increase the speed effectively
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
        return tensor

    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        # another solution is: torch.unsqueeze(tensor, dim=dim)
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def forward(self, x):
        # use the same variable naming as:
        # https://github.com/wentaoyuan/pcn/blob/master/models/pcn_cd.py
        feature = self.feat(x)

        coarse = self.folding1(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)

        grid = self.build_grid(x.shape[0])
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = point_feat.view([-1, self.num_fine, 3])

        global_feat = self.tile(self.expand_dims(feature, 1), [1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)

        center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.view([-1, self.num_fine, 3])

        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center

        return coarse, fine


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    @staticmethod
    def dist_cd(pc1, pc2):
        chamfer_dist = ChamferDistance()
        dist1, dist2 = chamfer_dist(pc1, pc2)
        return (torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2)))/2

    def forward(self, coarse, fine, gt, alpha):
        return self.dist_cd(coarse, gt) + alpha * self.dist_cd(fine, gt)


if __name__ == '__main__':

    model = get_model()
    print(model)
    input_pc = torch.rand(7, 3, 1024)
    x = model(input_pc)
