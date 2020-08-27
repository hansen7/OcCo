#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com
#  Ref: https://github.com/hansen7/NRS_3D/blob/master/train_dgcnn_cls.py
import os, sys, pdb, shutil, argparse, numpy as np, tensorflow as tf
from tqdm import tqdm
from termcolor import colored
from utils.Train_Logger import TrainLogger
from utils.Dataset_Assign import Dataset_Assign
# from utils.tf_util import get_bn_decay, get_lr_dgcnn
# from utils.io_util import shuffle_data, loadh5DataFile
# from utils.transfer_pretrained_w import load_pretrained_var
from utils.pc_util import random_point_dropout, random_scale_point_cloud, random_shift_point_cloud
from utils.ModelNetDataLoader import General_CLSDataLoader_HDF5
from torch.utils.data import DataLoader

def parse_args():
	parser = argparse.ArgumentParser(description='DGCNN Point Cloud Recognition Training Configuration')

	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--log_dir', default='occo_dgcnn_cls')
	parser.add_argument('--model', default='dgcnn_cls')
	parser.add_argument('--epoch', type=int, default=250)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--restore_path', type=str, default='')
	parser.add_argument('--batch_size', type=int, default=24)
	parser.add_argument('--num_points', type=int, default=1024)
	parser.add_argument('--base_lr', type=float, default=0.001)
	# parser.add_argument('--decay_steps', type=int, default=20)
	# parser.add_argument('--decay_rate', type=float, default=0.7)
	parser.add_argument('--momentum', type=float, default=0.9)

	parser.add_argument('--dataset', type=str, default='modelnet40')
	parser.add_argument('--filename', type=str, default='')
	parser.add_argument('--data_bn', action='store_true')
	parser.add_argument('--partial', action='store_true')
	parser.add_argument('--data_aug', action='store_true')
	parser.add_argument('--just_save', action='store_true')  # use only in the pretrained encoder restoration
	parser.add_argument('--fewshot', action='store_true')

	return parser.parse_args()


args = parse_args()

DATA_PATH = 'data/modelnet40_normal_resampled/'
NUM_CLASSES, NUM_TRAINOBJECTS, TRAIN_FILES, VALID_FILES = Dataset_Assign(
	dataset=args.dataset, fname=args.filename, partial=args.partial, bn=args.data_bn, few_shot=args.fewshot)
BATCH_SIZE, NUM_POINT = args.batch_size, args.num_points
# DECAY_STEP = NUM_TRAINOBJECTS//BATCH_SIZE * args.decay_steps

TRAIN_DATASET = General_CLSDataLoader_HDF5(file_list=TRAIN_FILES, num_point=1024)
TEST_DATASET = General_CLSDataLoader_HDF5(file_list=VALID_FILES, num_point=1024)
trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
testDataLoader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)
# reduce the num_workers if the loaded data are huge, ref: https://github.com/pytorch/pytorch/issues/973

def main(args):
	MyLogger = TrainLogger(args, name=args.model.upper(), subfold='log_cls')
	shutil.copy(os.path.join('cls_models', '%s.py' % args.model), MyLogger.log_dir)
	shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
	
	# is_training_pl -> to decide whether to apply batch normalisation
	is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
	global_step = tf.Variable(0, trainable=False, name='global_step')

	inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
	labels_pl = tf.placeholder(tf.int32, (BATCH_SIZE,), 'labels')
	npts_pl = tf.placeholder(tf.int32, (BATCH_SIZE,), 'num_points')

	# bn_decay = get_bn_decay(batch=global_step, bn_init_decay=0.5, batch_size=args.batch_size,
	# 						bn_decay_step=DECAY_STEP, bn_decay_rate=0.5, bn_decay_clip=0.99)

	bn_decay = 0.9
	# See "BatchNorm1d" in https://pytorch.org/docs/stable/nn.html
	''' === fix issues of importlib when running on some servers (i.e., woma) === '''
	# model_module = importlib.import_module('.%s' % args.model_type, 'cls_models')
	# MODEL = model_module.Model(inputs_pl, npts_pl, labels_pl, is_training_pl, bn_decay=bn_decay)
	ldic = locals()
	exec('from cls_models.%s import Model' % args.model, globals(), ldic)
	MODEL = ldic['Model'](inputs_pl, npts_pl, labels_pl, is_training_pl, bn_decay=bn_decay)
	pred, loss = MODEL.pred, MODEL.loss
	tf.summary.scalar('loss', loss)

	correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
	accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(args.batch_size)
	tf.summary.scalar('accuracy', accuracy)

	''' === Learning Rate === '''
	def get_lr_dgcnn(args, global_step, alpha):
		learning_rate = tf.train.cosine_decay(
			learning_rate=100 * args.base_lr,  # Base Learning Rate, 0.1
			global_step=global_step,  # Training Step Index
			decay_steps=NUM_TRAINOBJECTS//BATCH_SIZE * args.epoch,  # Total Training Step
			alpha=alpha  # Fraction of the Minimum Value of the Set lr
		)
		# learning_rate = tf.maximum(learning_rate, args.base_lr)
		return learning_rate

	learning_rate = get_lr_dgcnn(args, global_step, alpha=0.01)
	tf.summary.scalar('learning rate', learning_rate)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epoch, eta_min=args.lr)
	# doc: https://pytorch.org/docs/stable/optim.html
	# doc: https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay

	''' === Optimiser === '''
	# trainer = tf.train.GradientDescentOptimizer(learning_rate)
	trainer = tf.train.MomentumOptimizer(learning_rate, momentum=args.momentum)
	# equivalent to torch.optim.SGD
	# doc: https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/MomentumOptimizer
	# another alternative is to use keras
	# trainer = tf.keras.optimizers.SGD(learning_rate, momentum=args.momentum)
	# doc: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/optimizers/SGD
	# opt = torch.optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)

	train_op = trainer.minimize(loss=MODEL.loss, global_step=global_step)
	saver = tf.train.Saver()

	# ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
	config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True
	# config.allow_soft_placement = True  # Uncomment it if GPU option is not available
	# config.log_device_placement = True  # Uncomment it if you want device placements to be logged
	sess = tf.Session(config=config)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(os.path.join(MyLogger.experiment_dir, 'runs', 'train'), sess.graph)
	val_writer = tf.summary.FileWriter(os.path.join(MyLogger.experiment_dir, 'runs', 'valid'), sess.graph)

	# Initialise all the variables of the models
	init = tf.global_variables_initializer()

	sess.run(init, {is_training_pl: True})

	# to save the randomized initialised models then exit
	if args.just_save:
		save_path = saver.save(sess, os.path.join(MyLogger.checkpoints_dir, "model.ckpt"))
		print(colored('random initialised model saved at %s' % save_path, 'white', 'on_blue'))
		print(colored('just save the model, now exit', 'white', 'on_red'))
		sys.exit()

	'''current solution: first load pretrained encoder, 
	assemble with randomly initialised FC layers then save to the checkpoint'''

	if args.restore:
		saver.restore(sess, tf.train.latest_checkpoint(args.restore_path))
		MyLogger.logger.info('Model Parameters has been Restored')

	ops = {'pointclouds_pl': inputs_pl,
		   'labels_pl': labels_pl,
		   'is_training_pl': is_training_pl,
		   'npts_pl': npts_pl,
		   'pred': pred,
		   'loss': loss,
		   'train_op': train_op,
		   'merged': merged,
		   'step': global_step}

	for epoch in range(args.epoch):

		'''=== training the model ==='''
		train_one_epoch(sess, ops, MyLogger, train_writer)

		'''=== evaluating the model ==='''
		save_checkpoint = eval_one_epoch(sess, ops, MyLogger, val_writer)

		'''=== check whether to store the checkpoints ==='''
		if save_checkpoint:
			save_path = saver.save(sess, os.path.join(MyLogger.savepath, "model.ckpt"))
			MyLogger.logger.info('model saved at %s' % MyLogger.savepath)

	sess.close()
	MyLogger.train_summary()


def train_one_epoch(sess, ops, MyLogger, train_writer):
	is_training = True
	MyLogger.epoch_init(training=is_training)

	for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
		# pdb.set_trace()
		points, target = points.numpy(), target.numpy()

		if args.data_aug:
			points = random_point_dropout(points)
			points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
			points[:, :, 0:3] = random_shift_point_cloud(points[:, :, 0:3])

		feed_dict = {
			ops['pointclouds_pl']: points.reshape([1, BATCH_SIZE * NUM_POINT, 3]),
			ops['labels_pl']: target.reshape(BATCH_SIZE, ),
			ops['npts_pl']: [NUM_POINT] * BATCH_SIZE,
			ops['is_training_pl']: is_training}

		summary, step, _, loss, pred = sess.run([
			ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
		train_writer.add_summary(summary, step)

		# pdb.set_trace()
		MyLogger.step_update(np.argmax(pred, 1), target.reshape(BATCH_SIZE, ), loss)

	MyLogger.epoch_summary(writer=None, training=is_training)

	return None


def eval_one_epoch(sess, ops, MyLogger, val_writer):
	is_training = False
	MyLogger.epoch_init(training=is_training)

	for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
		# pdb.set_trace()
		points, target = points.numpy(), target.numpy()		

		feed_dict = {
				ops['pointclouds_pl']: points.reshape([1, BATCH_SIZE * NUM_POINT, 3]),
				ops['labels_pl']: target.reshape(BATCH_SIZE, ),
				ops['npts_pl']: np.array([NUM_POINT] * BATCH_SIZE),
				ops['is_training_pl']: is_training}

		summary, step, loss_val, pred_val = sess.run(
			[ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
		val_writer.add_summary(summary, step)
		# pdb.set_trace()
		MyLogger.step_update(np.argmax(pred_val, 1), target.reshape(BATCH_SIZE, ), loss_val)

	MyLogger.epoch_summary(writer=None, training=is_training)

	return MyLogger.save_model


if __name__ == '__main__':

	print('Now Using GPU:%s to train the model' % args.gpu)
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	main(args)
