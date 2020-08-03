#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import os, sys, pdb, time, argparse, datetime, importlib, numpy as np, tensorflow as tf
from tqdm import tqdm
from termcolor import colored
from utils.Dataset_Assign import Dataset_Assign
from utils.io_util import shuffle_data, loadh5DataFile
from utils.EarlyStoppingCriterion import EarlyStoppingCriterion
from utils.tf_util import add_train_summary, get_bn_decay, get_learning_rate
from utils.pc_util import rotate_point_cloud, jitter_point_cloud, random_point_dropout, \
	random_scale_point_cloud, random_shift_point_cloud

# from utils.transfer_pretrained_w import load_pretrained_var
from utils.ModelNetDataLoader import General_CLSDataLoader_HDF5
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

''' === Basic Learning Settings === '''
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--log_dir', default='log/log_cls/pointnet_cls')
parser.add_argument('--model', default='pointnet_cls')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--restore', action='store_true')
parser.add_argument('--restore_path', default='log/pointnet_cls')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_point', type=int, default=1024)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--lr_clip', type=float, default=1e-5)
parser.add_argument('--decay_steps', type=int, default=20)
parser.add_argument('--decay_rate', type=float, default=0.7)
# parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='modelnet40')
parser.add_argument('--partial', action='store_true')
parser.add_argument('--filename', type=str, default='')
parser.add_argument('--data_bn', action='store_true')

''' === Data Augmentation Settings === '''
parser.add_argument('--data_aug', action='store_true')
parser.add_argument('--just_save', action='store_true')  # pretrained encoder restoration
parser.add_argument('--patience', type=int, default=200)  # early stopping, set it as 200 for deprecation
parser.add_argument('--fewshot', action='store_true')

args = parser.parse_args()

DATA_PATH = 'data/modelnet40_normal_resampled/'
NUM_CLASSES, NUM_TRAINOBJECTS, TRAIN_FILES, VALID_FILES = Dataset_Assign(
	dataset=args.dataset, fname=args.filename, partial=args.partial, bn=args.data_bn, few_shot=args.fewshot)
TRAIN_DATASET = General_CLSDataLoader_HDF5(root=DATA_PATH, file_list=TRAIN_FILES, num_point=1024)
TEST_DATASET = General_CLSDataLoader_HDF5(root=DATA_PATH, file_list=VALID_FILES, num_point=1024)
trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
BASE_LR = args.base_lr
LR_CLIP = args.lr_clip
DECAY_RATE = args.decay_rate
# DECAY_STEP = args.decay_steps
DECAY_STEP = NUM_TRAINOBJECTS//BATCH_SIZE * args.decay_steps
BN_INIT_DECAY = 0.5
BN_DECAY_RATE = 0.5
BN_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
LOG_DIR = args.log_dir
BEST_EVAL_ACC = 0
os.system('mkdir -p %s' % LOG_DIR) if not os.path.exists(LOG_DIR) else None
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a+')

def log_string(out_str):
	LOG_FOUT.write(out_str + '\n')
	LOG_FOUT.flush()
	print(out_str)


def train(args):

	log_string('\n\n' + '=' * 50)
	log_string('Start Training, Time: %s' % datetime.datetime.now())
	log_string('=' * 50 + '\n\n')

	is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
	global_step = tf.Variable(0, trainable=False, name='global_step')  # will be used in defining train_op
	inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
	labels_pl = tf.placeholder(tf.int32, (BATCH_SIZE,), 'labels')
	npts_pl = tf.placeholder(tf.int32, (BATCH_SIZE,), 'num_points')

	bn_decay = get_bn_decay(global_step, BN_INIT_DECAY, BATCH_SIZE, BN_DECAY_STEP, BN_DECAY_RATE, BN_DECAY_CLIP)

	# model_module = importlib.import_module('.%s' % args.model, 'cls_models')
	# MODEL = model_module.Model(inputs_pl, npts_pl, labels_pl, is_training_pl, bn_decay=bn_decay)
	''' === To fix issues when running on woma === '''
	ldic = locals()
	exec('from cls_models.%s import Model' % args.model, globals(), ldic)
	MODEL = ldic['Model'](inputs_pl, npts_pl, labels_pl, is_training_pl, bn_decay=bn_decay)
	pred, loss = MODEL.pred, MODEL.loss
	tf.summary.scalar('loss', loss)
	# pdb.set_trace()

	# useful information in displaying during training
	correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
	accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
	tf.summary.scalar('accuracy', accuracy)

	learning_rate = get_learning_rate(global_step, BASE_LR, BATCH_SIZE, DECAY_STEP, DECAY_RATE, LR_CLIP)
	add_train_summary('learning_rate', learning_rate)
	trainer = tf.train.AdamOptimizer(learning_rate)
	train_op = trainer.minimize(MODEL.loss, global_step)
	saver = tf.train.Saver()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	# config.log_device_placement = True
	sess = tf.Session(config=config)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
	val_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'val'))

	# Init variables
	init = tf.global_variables_initializer()
	log_string('\nModel Parameters has been Initialized\n')
	sess.run(init, {is_training_pl: True})  # restore will cover the random initialized parameters

	# to save the randomized variables
	if not args.restore and args.just_save:
		save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
		print(colored('random initialised model saved at %s' % save_path, 'white', 'on_blue'))
		print(colored('just save the model, now exit', 'white', 'on_red'))
		sys.exit()

	'''current solution: first load pretrained head, assemble with output layers then save as a checkpoint'''
	# to partially load the saved head from:
	# if args.load_pretrained_head:
	#   sess.close()
	#   load_pretrained_head(args.pretrained_head_path, os.path.join(LOG_DIR, 'model.ckpt'), None, args.verbose)
	#   print('shared varibles have been restored from ', args.pretrained_head_path)
	#
	#   sess = tf.Session(config=config)
	#   log_string('\nModel Parameters has been Initialized\n')
	#   sess.run(init, {is_training_pl: True})
	#   saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
	#   log_string('\nModel Parameters have been restored with pretrained weights from %s' % args.pretrained_head_path)

	if args.restore:
		# load_pretrained_var(args.restore_path, os.path.join(LOG_DIR, "model.ckpt"), args.verbose)
		saver.restore(sess, tf.train.latest_checkpoint(args.restore_path))
		log_string('\n')
		log_string(colored('Model Parameters have been restored from %s' % args.restore_path, 'white', 'on_red'))

	for arg in sorted(vars(args)):
		print(arg + ': ' + str(getattr(args, arg)) + '\n')  # log of arguments
	os.system('cp cls_models/%s.py %s' % (args.model, LOG_DIR))  # bkp of model def
	os.system('cp train_cls.py %s' % LOG_DIR)  # bkp of train procedure

	train_start = time.time()

	ops = {'pointclouds_pl': inputs_pl,
		   'labels_pl': labels_pl,
		   'is_training_pl': is_training_pl,
		   'npts_pl': npts_pl,
		   'pred': pred,
		   'loss': loss,
		   'train_op': train_op,
		   'merged': merged,
		   'step': global_step}

	ESC = EarlyStoppingCriterion(patience=args.patience)

	for epoch in range(args.epoch):
		log_string('\n\n')
		log_string(colored('**** EPOCH %03d ****' % epoch, 'grey', 'on_green'))
		sys.stdout.flush()

		'''=== training the model ==='''
		train_one_epoch(sess, ops, train_writer)

		'''=== evaluating the model ==='''
		eval_mean_loss, eval_acc, eval_cls_acc = eval_one_epoch(sess, ops, val_writer)

		'''=== check whether to early stop ==='''
		early_stop, save_checkpoint = ESC.step(eval_acc, epoch=epoch)
		if save_checkpoint:
			save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
			log_string(colored('model saved at %s' % save_path, 'white', 'on_blue'))
		if early_stop:
			break

	log_string('total time: %s' % datetime.timedelta(seconds=time.time() - train_start))
	log_string('stop epoch: %d, best eval acc: %f' % (ESC.best_epoch + 1, ESC.best_dev_score))
	sess.close()


def train_one_epoch(sess, ops, train_writer):
	is_training = True
	total_correct, total_seen, loss_sum = 0, 0, 0

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

		summary, step, _, loss_val, pred_val = sess.run([
			ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
		train_writer.add_summary(summary, step)

		pred_val = np.argmax(pred_val, 1)
		correct = np.sum(pred_val == target.reshape(BATCH_SIZE, ))
		total_correct += correct
		total_seen += BATCH_SIZE
		# loss_sum += loss_val

	# train_file_idxs = np.arange(0, len(TRAIN_FILES))
	# np.random.shuffle(train_file_idxs)
	#
	# for fn in range(len(TRAIN_FILES)):
	# 	current_data, current_label = loadh5DataFile(TRAIN_FILES[train_file_idxs[fn]])
	# 	current_data = current_data[:, :NUM_POINT, :]
	# 	current_data, current_label, _ = shuffle_data(current_data, np.squeeze(current_label))
	# 	current_label = np.squeeze(current_label)
	#
	# 	file_size = current_data.shape[0]
	# 	num_batches = file_size // BATCH_SIZE
	#
	# 	for batch_idx in range(num_batches):
	# 		start_idx = batch_idx * BATCH_SIZE
	# 		end_idx = (batch_idx + 1) * BATCH_SIZE
	# 		feed_data = current_data[start_idx:end_idx, :, :]
	#
	# 		if args.data_aug:
	# 			feed_data = random_point_dropout(feed_data)
	# 			feed_data[:, :, 0:3] = random_scale_point_cloud(feed_data[:, :, 0:3])
	# 			feed_data[:, :, 0:3] = random_shift_point_cloud(feed_data[:, :, 0:3])
	#
	# 		feed_dict = {
	# 			ops['pointclouds_pl']: feed_data.reshape([1, BATCH_SIZE * NUM_POINT, 3]),
	# 			ops['labels_pl']: current_label[start_idx:end_idx].reshape(BATCH_SIZE, ),
	# 			ops['npts_pl']: [NUM_POINT] * BATCH_SIZE,
	# 			ops['is_training_pl']: is_training}
	#
	# 		summary, step, _, loss_val, pred_val = sess.run([
	# 			ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
	# 		train_writer.add_summary(summary, step)
	#
	# 		pred_val = np.argmax(pred_val, 1)
	# 		correct = np.sum(pred_val == current_label[start_idx:end_idx].reshape(BATCH_SIZE, ))
	# 		total_correct += correct
	# 		total_seen += BATCH_SIZE
	# 		loss_sum += loss_val

	log_string('\n=== training ===')
	log_string('total correct: %d, total_seen: %d' % (total_correct, total_seen))
	# log_string('mean batch loss: %f' % (loss_sum / num_batches))
	log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, val_writer):
	is_training = False

	total_correct, total_seen, loss_sum = 0, 0, 0
	total_seen_class = [0 for _ in range(NUM_CLASSES)]
	total_correct_class = [0 for _ in range(NUM_CLASSES)]

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
		pred_val = np.argmax(pred_val, 1)
		correct = np.sum(pred_val == target.reshape(BATCH_SIZE, ))
		total_correct += correct
		total_seen += BATCH_SIZE
		loss_sum += (loss_val * BATCH_SIZE)

		for i, l in enumerate(target):
			# l = int(target.reshape(-1)[i])
			# pdb.set_trace()
			total_seen_class[int(l)] += 1
			total_correct_class[int(l)] += (int(pred_val[i]) == int(l))

	# for fn in VALID_FILES:
	# 	current_data, current_label = loadh5DataFile(fn)
	# 	current_data = current_data[:, :NUM_POINT, :]
	# 	file_size = current_data.shape[0]
	# 	num_batches = file_size // BATCH_SIZE
	#
	# 	for batch_idx in range(num_batches):
	# 		start_idx, end_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
	#
	# 		feed_dict = {
	# 			ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :].reshape([1, BATCH_SIZE * NUM_POINT, 3]),
	# 			ops['labels_pl']: current_label[start_idx:end_idx].reshape(BATCH_SIZE, ),
	# 			ops['npts_pl']: np.array([NUM_POINT] * BATCH_SIZE),
	# 			ops['is_training_pl']: is_training}
	#
	# 		summary, step, loss_val, pred_val = sess.run(
	# 			[ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
	# 		val_writer.add_summary(summary, step)
	# 		pred_val = np.argmax(pred_val, 1)
	# 		correct = np.sum(pred_val == current_label[start_idx:end_idx].reshape(BATCH_SIZE, ))
	# 		total_correct += correct
	# 		total_seen += BATCH_SIZE
	# 		loss_sum += (loss_val * BATCH_SIZE)
	#
	# 		for i in range(start_idx, end_idx):
	# 			l = int(current_label.reshape(-1)[i])
	# 			total_seen_class[l] += 1
	# 			total_correct_class[l] += (pred_val[i - start_idx] == l)

	eval_mean_loss = loss_sum / float(total_seen)
	eval_acc = total_correct / float(total_seen)
	eval_cls_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
	log_string('\n=== evaluating ===')
	log_string('total correct: %d, total_seen: %d' % (total_correct, total_seen))
	log_string('eval mean loss: %f' % eval_mean_loss)
	log_string('eval accuracy: %f' % eval_acc)
	log_string('eval avg class acc: %f' % eval_cls_acc)

	global BEST_EVAL_ACC
	if eval_acc > BEST_EVAL_ACC:
		BEST_EVAL_ACC = eval_acc
	log_string('best eval accuracy: %f' % BEST_EVAL_ACC)
	return eval_mean_loss, eval_acc, eval_cls_acc


if __name__ == '__main__':
	print('Now Using GPU:%d to train the model' % args.gpu)
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

	train(args)
	LOG_FOUT.close()
