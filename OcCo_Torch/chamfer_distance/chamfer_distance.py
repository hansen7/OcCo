#  Ref: https://github.com/chrdiller/pyTorchChamferDistance
import os, torch, torch.nn as nn
from torch.utils.cpp_extension import load

basedir = os.path.dirname(__file__)
cd = load(name="cd", sources=[
	os.path.join(basedir, "chamfer_distance.cpp"),
	os.path.join(basedir, "chamfer_distance.cu")])

class ChamferDistanceFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, xyz1, xyz2):
		batchsize, n, _ = xyz1.size()
		_, m, _ = xyz2.size()
		xyz1 = xyz1.contiguous()
		xyz2 = xyz2.contiguous()
		dist1 = torch.zeros(batchsize, n)
		dist2 = torch.zeros(batchsize, m)

		idx1 = torch.zeros(batchsize, n, dtype=torch.int)
		idx2 = torch.zeros(batchsize, m, dtype=torch.int)

		if not xyz1.is_cuda:
			cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
		else:
			dist1 = dist1.cuda()
			dist2 = dist2.cuda()
			idx1 = idx1.cuda()
			idx2 = idx2.cuda()
			cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

		ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

		return dist1, dist2

	@staticmethod
	def backward(ctx, graddist1, graddist2):
		xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

		graddist1 = graddist1.contiguous()
		graddist2 = graddist2.contiguous()

		gradxyz1 = torch.zeros(xyz1.size())
		gradxyz2 = torch.zeros(xyz2.size())

		if not graddist1.is_cuda:
			cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
		else:
			gradxyz1 = gradxyz1.cuda()
			gradxyz2 = gradxyz2.cuda()
			cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

		return gradxyz1, gradxyz2


class ChamferDistance(nn.Module):
	def forward(self, xyz1, xyz2):
		return ChamferDistanceFunction.apply(xyz1, xyz2)


class get_model(nn.Module):
	def __init__(self, channel=3):
		super(get_model, self).__init__()

		self.conv1 = nn.Conv1d(channel, 128, 1)

	def forward(self, x):
		_, D, N = x.size()
		x = self.conv1(x)
		x = x.view(-1, 128, 1).repeat(1, 1, 3)
		return x


if __name__ == '__main__':

	import random, numpy as np

	'''Sanity Check on the Consistency with TensorFlow'''
	random.seed(100)
	np.random.seed(100)

	chamfer_dist = ChamferDistance()
	# model = get_model().to(torch.device("cuda"))
	# model.train()

	xyz1 = np.random.randn(32, 16384, 3).astype('float32')
	xyz2 = np.random.randn(32, 1024, 3).astype('float32')

	# pdb.set_trace()
	# pc1 = torch.randn(1, 100, 3).cuda().contiguous()
	# pc1_new = model(pc1.transpose(2, 1))
	# pc2 = torch.randn(1, 50, 3).cuda().contiguous()

	dist1, dist2 = chamfer_dist(torch.Tensor(xyz1), torch.Tensor(xyz2))
	loss = (torch.mean(dist1)) + (torch.mean(dist2))
	print(loss)
	# loss.backward()
