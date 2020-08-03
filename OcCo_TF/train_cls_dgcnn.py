#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com
#  Ref: https://github.com/hansen7/NRS_3D/blob/master/train_dgcnn_cls.py
import os, sys, shutil, argparse, numpy as np, tensorflow as tf
from termcolor import colored
from utils.Train_Logger import TrainLogger
from utils.Dataset_Assign import DataSet_Assign
from utils.tf_util import get_bn_decay, get_lr_dgcnn
from utils.io_util import shuffle_data, loadh5DataFile
from utils.pc_util import random_point_dropout, random_scale_point_cloud, random_shift_point_cloud
# from utils.transfer_pretrained_w import load_pretrained_var

def parse_args():
	parser = argparse.ArgumentParser(description='Point Cloud Recognition Training Configuration')

	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--log_dir', default='occo_dgcnn_cls')
	parser.add_argument('--model', default='dgcnn_cls')
	parser.add_argument('--epoch', type=int, default=200)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--restore_path', type=str)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_points', type=int, default=1024)
	parser.add_argument('--base_lr', type=float, default=0.1)
	parser.add_argument('--decay_steps', type=int, default=200000)
	parser.add_argument('--decay_rate', type=float, default=0.7)
	parser.add_argument('--dataset', type=str, default='modelnet40')
	# TODO: to remove this in the public version
	parser.add_argument('--dataset_file', type=str, default=None, help='for scanobjectnn')
	parser.add_argument('--data_bn', action='store_true')
	parser.add_argument('--partial', action='store_true')
	parser.add_argument('--data_aug', action='store_true')
	parser.add_argument('--just_save', action='store_true')  # use only in the pretrained encoder restoration

	return parser.parse_args()


args = parse_args()
BATCH_SIZE = args.batch_size
NUM_POINT = args.num_points
NUM_CLASSES, TRAIN_FILES, VALID_FILES = DataSet_Assign(args.dataset, args.dataset_file, args.partial, args.data_bn)


def main(args):
	MyLogger = TrainLogger(args, name=args.model.upper(), subfold='classification')
	shutil.copy(os.path.join('cls_models', '%s.py' % args.model), MyLogger.log_dir)
	shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)

	is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
	global_step = tf.Variable(0, trainable=False, name='global_step')

	inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
	labels_pl = tf.placeholder(tf.int32, (BATCH_SIZE,), 'labels')
	npts_pl = tf.placeholder(tf.int32, (BATCH_SIZE,), 'num_points')

	bn_decay = get_bn_decay(batch=global_step, bn_init_decay=0.5, batch_size=args.batch_size,
							bn_decay_step=args.decap_steps, bn_decay_rate=0.5, bn_decay_clip=0.99)

	''' === fix issues of importlib when running on some servers (i.e., woma) === '''
	# model_module = importlib.import_module('.%s' % args.model_type, 'cls_models')
	# MODEL = model_module.Model(inputs_pl, npts_pl, labels_pl, is_training_pl, bn_decay=bn_decay)
	ldic = locals()
	exec('from cls_models.%s import Model' % args.model_type, globals(), ldic)
	MODEL = ldic['Model'](inputs_pl, npts_pl, labels_pl, is_training_pl, bn_decay=bn_decay)
	pred, loss = MODEL.pred, MODEL.loss
	tf.summary.scalar('loss', loss)

	correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
	accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(args.batch_sizes)
	tf.summary.scalar('accuracy', accuracy)

	learning_rate = get_lr_dgcnn(global_step, args.base_lr, args.batch_size,
								 args.decay_steps, args.decay_rate)
	# trainer = tf.train.GradientDescentOptimizer(learning_rate)
	trainer = tf.train.MomentumOptimizer(learning_rate, momentum=args.momentum)
	# opt = torch.optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
	train_op = trainer.minimize(MODEL.loss, global_step)
	saver = tf.train.Saver()

	# ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	# config.allow_soft_placement = True  # Uncomment it if GPU option is not available
	# config.log_device_placement = True  # Uncomment it if you want device placements to be logged
	sess = tf.Session(config=config)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(os.path.join(MyLogger.experiment_dir, 'runs', 'train'), sess.graph)
	val_writer = tf.summary.FileWriter(os.path.join(MyLogger.experiment_dir, 'runs', 'valid'), sess.graph)

	# Initialise all the variables of the models
	init = tf.global_variables_initializer()
	MyLogger.logger.info('Model Parameters has been Initialized')
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
			MyLogger.logger.info(colored('model saved at %s' % MyLogger.savepath, 'white', 'on_blue'))

	sess.close()
	MyLogger.cls_train_summary()


def train_one_epoch(sess, ops, MyLogger, train_writer):
	is_training = True
	MyLogger.cls_epoch_init(training=is_training)

	train_file_idxs = np.arange(0, len(TRAIN_FILES))
	np.random.shuffle(train_file_idxs)

	for fn in range(len(TRAIN_FILES)):
		current_data, current_label = loadh5DataFile(TRAIN_FILES[train_file_idxs[fn]])
		current_data = current_data[:, :NUM_POINT, :]
		current_data, current_label, _ = shuffle_data(current_data, np.squeeze(current_label))
		current_label = np.squeeze(current_label)

		file_size = current_data.shape[0]
		num_batches = file_size // BATCH_SIZE

		for batch_idx in range(num_batches):
			start_idx = batch_idx * BATCH_SIZE
			end_idx = (batch_idx + 1) * BATCH_SIZE
			feed_data = current_data[start_idx:end_idx, :, :]

			if args.data_aug:
				feed_data = random_point_dropout(feed_data)
				feed_data[:, :, 0:3] = random_scale_point_cloud(feed_data[:, :, 0:3])
				feed_data[:, :, 0:3] = random_shift_point_cloud(feed_data[:, :, 0:3])

			feed_dict = {
				ops['pointclouds_pl']: feed_data.reshape([1, args.batch_size * args.num_points, 3]),
				ops['labels_pl']: current_label[start_idx:end_idx].reshape(args.batch_size, ),
				ops['npts_pl']: [args.num_points] * args.batch_size,
				ops['is_training_pl']: is_training}

			summary, step, _, loss, pred = sess.run([
				ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
			train_writer.add_summary(summary, step)

			MyLogger.cls_step_update(np.argmax(pred, 1),
									 current_label[start_idx:end_idx].reshape(args.batch_size, ),
									 loss)

	MyLogger.cls_epoch_summary(writer=None, training=is_training)

	return None


def eval_one_epoch(sess, ops, MyLogger, val_writer):
	is_training = False
	MyLogger.cls_epoch_init(training=is_training)

	for fn in VALID_FILES:
		current_data, current_label = loadh5DataFile(fn)
		current_data = current_data[:, :NUM_POINT, :]
		file_size = current_data.shape[0]
		num_batches = file_size // BATCH_SIZE

		for batch_idx in range(num_batches):
			start_idx = batch_idx * BATCH_SIZE
			end_idx = (batch_idx + 1) * BATCH_SIZE

			feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :].reshape(
				[1, args.batch_size * args.num_points, 3]),
				ops['labels_pl']: current_label[start_idx:end_idx].reshape(args.batch_size, ),
				ops['npts_pl']: np.array([args.num_points] * args.batch_size),
				ops['is_training_pl']: is_training}

			summary, step, loss_val, pred_val = sess.run(
				[ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
			val_writer.add_summary(summary, step)
			MyLogger.cls_step_update(np.argmax(pred_val, 1),
									 current_label[start_idx:end_idx].reshape(args.batch_size, ),
									 loss_val)

	MyLogger.cls_epoch_summary(writer=None, training=is_training)

	return MyLogger.save_model


if __name__ == '__main__':

	print('Now Using GPU:%d to train the model' % args.gpu)
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	main(args)
