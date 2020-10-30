#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import pdb, torch, torch.nn as nn, torch.nn.functional as F
from pcn_util import PCNEncoder

class get_model(nn.Module):
	def __init__(self, num_class=40, num_channel=3, **kwargs):
		super(get_model, self).__init__()
		self.feat = PCNEncoder(global_feat=True, channel=num_channel)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, num_class)

		self.dp1 = nn.Dropout(p=0.3)
		self.bn1 = nn.BatchNorm1d(512)
		self.dp2 = nn.Dropout(p=0.3)
		self.bn2 = nn.BatchNorm1d(256)

	def forward(self, x):
		x = self.feat(x)
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.dp1(x)
		
		x = F.relu(self.bn2(self.fc2(x)))
		x = self.dp2(x)

		x = self.fc3(x)
		x = F.log_softmax(x, dim=1)
		return x


class get_loss(nn.Module):
	def __init__(self):
		super(get_loss, self).__init__()

	def forward(self, pred, target):
		loss = F.nll_loss(pred, target)
		return loss


if __name__ == '__main__':

	model = get_model()
	xyz = torch.rand(12, 3, 1024)
	x = model(xyz)
