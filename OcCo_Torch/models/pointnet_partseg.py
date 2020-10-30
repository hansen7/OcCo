#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_part_seg.py

import torch.nn as nn, torch.nn.functional as F
from pointnet_util import PointNetPartSegEncoder, feature_transform_regularizer


class get_model(nn.Module):
    def __init__(self, part_num=50, num_channel=3, **kwargs):
        super(get_model, self).__init__()
        self.part_num = part_num
        self.feat = PointNetPartSegEncoder(feature_transform=True,
                                           channel=num_channel)

        self.convs1 = nn.Conv1d(4944, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        concat, trans_feat = self.feat(point_cloud, label)

        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net).transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num)  # [B, N, 50]

        return net, trans_feat


class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

