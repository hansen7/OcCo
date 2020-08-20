#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_cls.py

import torch.nn as nn, torch.nn.functional as F
from pointnet_util import PointNetEncoder, feature_transform_regularizer


class get_model(nn.Module):
	def __init__(self, num_class=40, num_channel=3, **kwargs):
		super(get_model, self).__init__()
		self.feat = PointNetEncoder(
			global_feat=True, feature_transform=True, channel=num_channel)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, num_class)
		self.dropout = nn.Dropout(p=0.3)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)

	def forward(self, x):
		x, trans, trans_feat = self.feat(x)
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.dropout(self.fc2(x))))
		x = self.fc3(x)
		x = F.log_softmax(x, dim=1)
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
