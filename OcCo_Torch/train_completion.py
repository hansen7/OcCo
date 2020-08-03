#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/wentaoyuan/pcn/blob/master/train.py
#  Ref: https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/train.py
#  For DGCNN Feature Encoder, We Use Adam + StepLR for the Unity and Simplicity

import os, sys, pdb, time, torch, shutil, argparse, datetime, importlib, numpy as np
sys.path.append('utils')
sys.path.append('models')
# from tqdm import tqdm
from TrainLogger import TrainLogger
from LMDB_DataFlow import lmdb_dataflow
from Torch_Utility import copy_parameters
# from torch.optim.lr_scheduler import StepLR
from Visu_Utility import plot_pcd_three_views
from torch.utils.tensorboard import SummaryWriter


def parse_args():
	parser = argparse.ArgumentParser('Point Cloud Completion')

	''' === Training Setting === '''
	parser.add_argument('--log_dir', type=str, help='log folder [default: ]')
	parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size [default: 32]')
	parser.add_argument('--epoch', type=int, default=50, help='number of epoch [default: 50]')
	parser.add_argument('--lr', type=float, default=0.0001, help='learning rate [default: 1e-4]')
	parser.add_argument('--lr_decay', type=float, default=0.7, help='lr decay rate [default: 0.7]')
	parser.add_argument('--step_size', type=int, default=20, help='lr decay step [default: 20 epoch]')
	parser.add_argument('--dataset', type=str, default='modelnet', help='dataset [default: modelnet]')
	parser.add_argument('--restore', action='store_true', help='go from pre-trained [default: False]')
	parser.add_argument('--restore_path', type=str, help='path to pre-trained weights [default: None]')
	parser.add_argument('--epochs_save', type=int, default=5, help='number of epochs to save [default: 5]')
	parser.add_argument('--steps_print', type=int, default=100, help='number of steps to print [default: 1e2]')
	parser.add_argument('--steps_visu', type=int, default=3456, help='number of steps to visual [default: 3456]')
	parser.add_argument('--steps_eval', type=int, default=1000, help='number of steps to evaluate [default: 1e3]')

	''' === Model Setting === '''
	parser.add_argument('--model', type=str, default='pointnet_occo', help='model [pointnet_occo]')
	parser.add_argument('--grid_size', type=int, default=4, help='side length of 2D grid plane[4]')
	parser.add_argument('--grid_scale', type=float, default=1024, help='scale of the 2D grid [0.5]')
	parser.add_argument('--num_coarse', type=int, default=1024, help='points number in coarse [1024]')
	parser.add_argument('--k', type=int, default=20, help='number of nearest neighbors in DGCNN [20]')
	parser.add_argument('--emb_dims', type=int, default=1024, help='dimension of DGCNN embedding [1024]')
	parser.add_argument('--input_pts', type=int, default=1024, help='points number of occluded objects [1024]')
	parser.add_argument('--gt_pts', type=int, default=16384, help='points number of fully observed object [16384]')

	return parser.parse_args()


def main(args):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	''' === Set up Loggers and Load Data === '''
	MyLogger = TrainLogger(args, name=args.model.upper(), subfold='completion')
	os.makedirs(os.path.join(MyLogger.experiment_dir, 'plots'), exist_ok=True)
	writer = SummaryWriter(os.path.join(MyLogger.experiment_dir, 'runs'))

	MyLogger.logger.info('Load dataset %s' % args.dataset)
	if args.dataset == 'modelnet':
		lmdb_train = './data/modelnet/train.lmdb'
		lmdb_valid = './data/modelnet/test.lmdb'
	elif args.dataset == 'shapenet':
		lmdb_train = 'data/shapenet/train.lmdb'
		lmdb_valid = 'data/shapenet/valid.lmdb'
	else:
		raise ValueError("Dataset is not available, it should be either ModelNet or ShapeNet")

	assert (args.gt_pts == args.grid_size ** 2 * args.num_coarse)
	df_train, num_train = lmdb_dataflow(
		lmdb_train, args.batch_size, args.input_pts, args.gt_pts, is_training=True)
	train_gen = df_train.get_data()
	df_valid, num_valid = lmdb_dataflow(
		lmdb_valid, args.batch_size, args.input_pts, args.gt_pts, is_training=False)
	valid_gen = df_valid.get_data()

	''' === Load Model and Backup Scripts === '''
	MODEL = importlib.import_module(args.model)
	shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
	shutil.copy('./models/%s.py' % args.model, MyLogger.log_dir)

	# allow multiple GPUs
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	completer = MODEL.get_model(args=args).to(device)
	criterion = MODEL.get_loss().to(device)
	completer = torch.nn.DataParallel(completer)
	# nn.DataParallel has its own issues (slow, memory expensive),
	# here are some advanced solutions: https://zhuanlan.zhihu.com/p/145427849
	print('=' * 33)
	print('Using %d GPU,' % torch.cuda.device_count(), 'Indices are: %s' % args.gpu)
	print('=' * 33)

	''' === Restore Model from Checkpoints, If there is any === '''
	if args.restore:
		checkpoint = torch.load(args.restore_path)
		completer = copy_parameters(completer, checkpoint, verbose=True)
		MyLogger.logger.info('Use pre-trained model from %s' % args.restore_path)
		MyLogger.step, MyLogger.epoch = checkpoint['step'], checkpoint['epoch']

	else:
		MyLogger.logger.info('No pre-trained model, start training from scratch...')

	optimizer = torch.optim.Adam(
		completer.parameters(),
		lr=args.lr,
		betas=(0.9, 0.999),
		eps=1e-08,
		weight_decay=1e-4)

	# For the sake of simplicity, we omit the momentum decay in the batch norm
	# scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
	LEARNING_RATE_CLIP = 0.01 * args.lr

	def vary2fix(inputs, npts, batch_size=args.batch_size, num_point=args.input_pts):
		"""upsample/downsample varied input points into fixed length
		:param inputs: input points cloud
		:param npts: describe how many points of each input object
		:param batch_size: training batch size
		:param num_point: number of points of per occluded object
		:return: fixed length of points of each object
		"""

		inputs_ls = np.split(inputs[0], npts.cumsum())
		ret_inputs = np.zeros((1, batch_size * num_point, 3))
		ret_npts = npts.copy()

		for idx, obj in enumerate(inputs_ls[:-1]):

			if len(obj) <= num_point:
				select_idx = np.concatenate([
					np.arange(len(obj)), np.random.choice(len(obj), num_point - len(obj))])
			else:
				select_idx = np.arange(len(obj))
				np.random.shuffle(select_idx)

			ret_inputs[0][idx * num_point:(idx + 1) * num_point] = obj[select_idx].copy()
			ret_npts[idx] = num_point

		return ret_inputs, ret_npts

	def piecewise_constant(global_step, boundaries, values):
		"""substitute for tf.train.piecewise_constant:
		https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/piecewise_constant
		global_step can be either training epoch or training step
		"""
		if len(boundaries) != len(values) - 1:
			raise ValueError(
				"The length of boundaries should be 1 less than the length of values")

		if global_step <= boundaries[0]:  # right continuous
			return values[0]
		elif global_step > boundaries[-1]:
			return values[-1]
		else:
			for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
				if (global_step > low) & (global_step <= high):
					return v

	total_steps = num_train // args.batch_size * args.epoch
	total_time, train_start = 0, time.time()

	for step in range(MyLogger.step + 1, total_steps + 1):

		''' === Training === '''
		start = time.time()
		epoch = step * args.batch_size // num_train + 1
		lr = max(args.lr * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		# this is the original alpha setting for ShapeNet Dataset in PCN paper:
		alpha = piecewise_constant(step, [10000, 20000, 50000], [0.01, 0.1, 0.5, 1.0])
		writer.add_scalar('Learning Rate', lr, step)
		writer.add_scalar('Alpha', alpha, step)

		ids, inputs, npts, gt = next(train_gen)
		if args.dataset == 'shapenet':
			inputs, _ = vary2fix(inputs, npts)

		completer.train()
		optimizer.zero_grad()
		inputs = inputs.reshape(args.batch_size, args.input_pts, 3)
		inputs, gt = torch.Tensor(inputs).transpose(2, 1).cuda(), torch.Tensor(gt).cuda()
		pred_coarse, pred_fine = completer(inputs)
		loss = criterion(pred_coarse, pred_fine, gt, alpha)
		loss.backward()
		optimizer.step()
		total_time += time.time() - start

		if step % args.steps_print == 0:
			MyLogger.logger.info('epoch %d  step %d  alpha %.2f, loss %.8f - time per minibatch %.2f s' %
								 (epoch, step, alpha, loss, total_time / args.steps_print))
			total_time = 0

		''' === Validating === '''
		if step % args.steps_eval == 0:

			with torch.no_grad():
				completer.eval()
				MyLogger.logger.info('Testing...')

				num_eval_steps = num_valid // args.batch_size
				eval_loss, eval_time = 0, 0
				for eval_step in range(num_eval_steps):
					start = time.time()
					_, inputs, npts, gt = next(valid_gen)
					if args.dataset == 'shapenet':
						inputs, _ = vary2fix(inputs, npts)

					inputs = inputs.reshape(args.batch_size, args.input_pts, 3)
					inputs, gt = torch.Tensor(inputs).transpose(2, 1).cuda(), torch.Tensor(gt).cuda()

					pred_coarse, pred_fine = completer(inputs)
					loss = criterion(pred_coarse, pred_fine, gt, alpha)
					eval_loss += loss
					eval_time += time.time() - start

				MyLogger.logger.info('epoch %d  step %d  test loss %.8f - time per minibatch %.2f s' %
									 (epoch, step, eval_loss / num_eval_steps, eval_time / num_eval_steps))

		''' === Visualisation === '''
		if step % args.steps_visu == 0:
			all_pcds = [item.detach().cpu().numpy() for item in [
				inputs.transpose(2, 1), pred_coarse, pred_fine, gt]]
			for i in range(args.batch_size):
				# print(epoch, step, ids)	
				plot_path = os.path.join(MyLogger.experiment_dir, 'plots',
										 'epoch_%d_step_%d_%s.png' % (epoch, step, ids[i]))
				pcds = [x[i] for x in all_pcds]
				# pdb.set_trace()	
				plot_pcd_three_views(plot_path, pcds,
									 ['input', 'coarse output', 'fine output', 'ground truth'])

		if (epoch % args.epochs_save == 0) and \
			not os.path.exists(os.path.join(MyLogger.checkpoints_dir, 'model_epoch_%d.pth' % epoch)):
			# pdb.set_trace()
			state = {
				'step': step,
				'epoch': epoch,
				'model_state_dict': completer.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
			}
			torch.save(state, os.path.join(MyLogger.checkpoints_dir, "model_epoch_%d.pth" % epoch))
			MyLogger.logger.info('Model saved at %s/model_epoch_%d.pth\n' % (MyLogger.checkpoints_dir, epoch))

	MyLogger.logger.info('Training Finished, Total Time: ',
						 datetime.timedelta(seconds=time.time() - train_start))


if __name__ == '__main__':
	args = parse_args()
	main(args)
