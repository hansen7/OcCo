#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch, torch.nn as nn, torch.nn.functional as F
from pointnet_util import PointNetEncoder, feature_transform_regularizer


class get_model(nn.Module):
    def __init__(self, num_class, num_channel=3, **kwargs):
        super(get_model, self).__init__()
        self.num_class = num_class
        self.feat = PointNetEncoder(global_feat=False,
                                    feature_transform=True,
                                    channel=num_channel)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.num_class, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.num_class), dim=-1)
        x = x.view(batch_size, num_points, self.num_class)
        return x, trans_feat


class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    model = get_model(num_class=13, num_channel=3)
    xyz = torch.rand(12, 3, 2048)
    model(xyz)
