#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py

import torch

def knn(x, k):
	inner = -2 * torch.matmul(x.transpose(2, 1), x)
	xx = torch.sum(x ** 2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1)
	idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
	return idx


def get_graph_feature(x, k=20, idx=None, extra_dim=False):

	batch_size, num_dims, num_points = x.size()
	x = x.view(batch_size, -1, num_points)
	if idx is None:
		if extra_dim is False:
			idx = knn(x, k=k)
		else:
			# idx = knn(x[:, :3], k=k)
			idx = knn(x[:, 6:], k=k)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
	idx += idx_base
	idx = idx.view(-1)

	x = x.transpose(2, 1).contiguous()
	feature = x.view(batch_size*num_points, -1)[idx, :]
	feature = feature.view(batch_size, num_points, k, num_dims)
	x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
	feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

	return feature  # (batch_size, 2 * num_dims, num_points, k)
