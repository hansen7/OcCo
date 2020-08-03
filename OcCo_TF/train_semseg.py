#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import os, sys, pdb, shutil, socket, argparse, numpy as np, tensorflow as tf
from utils.io_util import getDataFiles, loadh5DataFile, shuffle_data
from utils.tf_util import get_bn_decay, get_learning_rate
from utils.Train_Logger import TrainLogger
from termcolor import colored
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'semseg_models'))

parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--log_dir', default='pointnet_semseg')
parser.add_argument('--num_point', type=int, default=4096)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--lr_clip', type=float, default=1e-5)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--decay_steps', type=int, default=300000)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--test_area', type=int, default=6)
parser.add_argument('--just_save', action='store_true', default=False)
parser.add_argument('--restore', action='store_true', default=False)
parser.add_argument('--restore_path', default='')
parser.add_argument('--model', default='pointnet_semseg')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_RATE = 0.5
BN_DECAY_STEP = float(args.decay_steps)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

ALL_FILES = getDataFiles(r'./data/indoor3d_sem_seg_hdf5_data/all_files.txt')
room_filelist = [line.rstrip() for line in open(r'./data/indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

# Load ALL data
data_batch_list, label_batch_list = [], []
for h5_filename in ALL_FILES:
	data_batch, label_batch = loadh5DataFile(h5_filename)
	data_batch_list.append(data_batch)
	label_batch_list.append(label_batch)

data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
test_area = 'Area_'+str(args.test_area)
train_idxs, test_idxs = [], []

for i, room_name in enumerate(room_filelist):
	if test_area in room_name:
		test_idxs.append(i)
	else:
		train_idxs.append(i)

train_data = data_batches[train_idxs, ...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs, ...]
test_label = label_batches[test_idxs]
print("Training Dataset Data and Label Shape:", train_data.shape, train_label.shape)
print("Testing Dataset Data and Label Shape:", test_data.shape, test_label.shape)

def main(args):

	MyLogger = TrainLogger(args, name=args.model.upper(), subfold='log_semseg')
	shutil.copy(os.path.join('semseg_models', '%s.py' % args.model), MyLogger.log_dir)
	shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)

	inputs_pl = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_POINT, 9), 'inputs')
	labels_pl = tf.placeholder(tf.int32, (BATCH_SIZE, NUM_POINT), 'labels')
	
	is_training_pl = tf.placeholder(tf.bool, shape=())
	global_step = tf.Variable(0, trainable=False, name='global_step')
	
	bn_decay = get_bn_decay(
		global_step, BN_INIT_DECAY, BATCH_SIZE, BN_DECAY_STEP, BN_DECAY_RATE, BN_DECAY_CLIP)
	
	# load model and loss
	ldic = locals()
	exec('from semseg_models.%s import Model' % args.model, globals(), ldic)
	MODEL = ldic['Model'](inputs_pl, labels_pl, is_training_pl, bn_decay=bn_decay)
	pred, loss = MODEL.pred, MODEL.loss
	tf.summary.scalar('loss', loss)
	
	correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
	accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
	tf.summary.scalar('accuracy', accuracy)
	
	learning_rate = get_learning_rate(
		global_step, args.base_lr, args.batch_size, args.decay_steps, args.decay_rate, args.lr_clip)
	tf.summary.scalar('learning_rate', learning_rate)
	if args.optimizer == 'momentum':
		optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=args.lr_clip)
	elif args.optimizer == 'adam':
		optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.minimize(loss, global_step=global_step)
	
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	
	# Create a session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	# config.allow_soft_placement = True
	# config.log_device_placement = True
	
	sess = tf.Session(config=config)
	
	# Add summary writers
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(os.path.join(MyLogger.experiment_dir, 'runs', 'train'), sess.graph)
	test_writer = tf.summary.FileWriter(os.path.join(MyLogger.experiment_dir, 'runs', 'valid'), sess.graph)

	# Init variables
	init = tf.global_variables_initializer()
	MyLogger.logger.info('Model Parameters has been Initialized')
	sess.run(init, {is_training_pl: True})

	if args.just_save:
		save_path = saver.save(sess, os.path.join(MyLogger.checkpoints_dir, "model.ckpt"))
		print(colored('random initialised model saved at %s' % save_path, 'white', 'on_blue'))
		print(colored('just save the model, now exit', 'white', 'on_red'))
		sys.exit()
	
	if args.restore:
		# load_pretrained_var(args.restore_path, os.path.join(LOG_DIR, "model.ckpt"), args.verbose)
		saver.restore(sess, tf.train.latest_checkpoint(args.restore_path))
		MyLogger.logger.info('Model Parameters has been Restored')
	
	ops = {'pointclouds_pl': inputs_pl,
	       'labels_pl': labels_pl,
	       'is_training_pl': is_training_pl,
	       'pred': pred,
	       'loss': loss,
	       'train_op': train_op,
	       'merged': merged,
	       'step': global_step}
	
	# ESC = EarlyStoppingCriterion(patience=args.patience)
	
	for epoch in range(args.epoch):

		'''=== training the model ==='''
		train_one_epoch(sess, ops, MyLogger, train_writer)

		'''=== evaluating the model ==='''
		save_checkpoint = eval_one_epoch(sess, ops, MyLogger, test_writer)
		
		'''=== check whether to early stop ==='''
		# early_stop, save_checkpoint = ESC.step(eval_acc, epoch=epoch)
		
		if save_checkpoint:
			save_path = saver.save(sess, os.path.join(MyLogger.savepath, "model.ckpt"))
			MyLogger.logger.info('model saved at %s' % MyLogger.savepath)

	sess.close()
	MyLogger.train_summary()

def train_one_epoch(sess, ops, MyLogger, train_writer):

	is_training = True
	MyLogger.epoch_init(training=is_training)

	current_data, current_label, _ = shuffle_data(train_data[:, 0:NUM_POINT, :], train_label)
	file_size = current_data.shape[0]
	num_batches = file_size // BATCH_SIZE
	
	for batch_idx in tqdm(range(num_batches)):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx+1) * BATCH_SIZE

		feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
		             ops['labels_pl']: current_label[start_idx:end_idx],
		             ops['is_training_pl']: is_training, }
		summary, step, _, loss, pred = sess.run(
			[ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
		train_writer.add_summary(summary, step)
		MyLogger.step_update(
			list(np.argmax(pred, 2).ravel()), 
			list(np.ravel(current_label[start_idx:end_idx])), 
			loss)
	
	# pdb.set_trace()
	MyLogger.seg_epoch_summary(writer=None, training=is_training)

	return None


def eval_one_epoch(sess, ops, MyLogger, test_writer):

	is_training = False
	MyLogger.epoch_init(training=is_training)

	current_data = test_data[:, 0:NUM_POINT, :]
	current_label = np.squeeze(test_label)
	file_size = current_data.shape[0]
	num_batches = file_size // BATCH_SIZE
	
	for batch_idx in tqdm(range(num_batches)):

		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx+1) * BATCH_SIZE
		feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
		             ops['labels_pl']: current_label[start_idx:end_idx],
		             ops['is_training_pl']: is_training}
		summary, step, loss_val, pred_val = sess.run(
			[ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
		test_writer.add_summary(summary, step)

		MyLogger.step_update(
			list(np.argmax(pred_val, 2).ravel()), 
			list(np.ravel(current_label[start_idx:end_idx])), 
			loss_val)

	MyLogger.seg_epoch_summary(writer=None, training=is_training)

	return MyLogger.save_model


if __name__ == "__main__":
	
	print('Now Using GPU:%s to train the model' % args.gpu)
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	
	main(args)

