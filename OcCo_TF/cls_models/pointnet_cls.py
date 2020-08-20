#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import sys, os
import tensorflow as tf
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from utils.tf_util import fully_connected, dropout, conv2d, max_pool2d
from train_cls import NUM_CLASSES, BATCH_SIZE, NUM_POINT
from utils.transform_nets import input_transform_net, feature_transform_net


class Model:
	def __init__(self, inputs, npts, labels, is_training, **kwargs):
		self.__dict__.update(kwargs)  # batch_decay and is_training
		self.is_training = is_training
		self.features = self.create_encoder(inputs, npts)
		self.pred = self.create_decoder(self.features)
		self.loss = self.create_loss(self.pred, labels)

	def create_encoder(self, inputs, npts):
		"""PointNet encoder"""
		
		inputs = tf.reshape(inputs, (BATCH_SIZE, NUM_POINT, 3))
		with tf.variable_scope('transform_net1') as sc:
			transform = input_transform_net(inputs, self.is_training, self.bn_decay, K=3)
		
		point_cloud_transformed = tf.matmul(inputs, transform)
		input_image = tf.expand_dims(point_cloud_transformed, -1)
		
		net = conv2d(inputs=input_image, num_output_channels=64, kernel_size=[1, 3],
		             scope='conv1', padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, bn_decay=self.bn_decay)
		net = conv2d(inputs=net, num_output_channels=64, kernel_size=[1, 1],
		             scope='conv2', padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, bn_decay=self.bn_decay)
		
		with tf.variable_scope('transform_net2') as sc:
			transform = feature_transform_net(net, self.is_training, self.bn_decay, K=64)
		net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
		net_transformed = tf.expand_dims(net_transformed, [2])
		
		'''conv2d, with kernel size of [1,1,1,1] and stride of [1,1,1,1],
		basically equals with the MLPs'''
		
		# use_xavier=True, stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
		net = conv2d(net_transformed, 64, [1, 1],
		             scope='conv3', padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, bn_decay=self.bn_decay)
		net = conv2d(net, 128, [1, 1],
		             padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training,
		             scope='conv4', bn_decay=self.bn_decay)
		net = conv2d(net, 1024, [1, 1],
		             padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training,
		             scope='conv5', bn_decay=self.bn_decay)
		
		net = max_pool2d(net, [NUM_POINT, 1],
		                 padding='VALID', scope='maxpool')
		
		features = tf.reshape(net, [BATCH_SIZE, -1])
		return features
	
	def create_decoder(self, features):
		"""fully connected layers for classification with dropout"""
		
		with tf.variable_scope('decoder_cls', reuse=tf.AUTO_REUSE):
			
			features = fully_connected(features, 512, bn=True, scope='fc1', is_training=self.is_training)
			features = dropout(features, keep_prob=0.7, scope='dp1', is_training=self.is_training)
			features = fully_connected(features, 256, bn=True, scope='fc2', is_training=self.is_training)
			features = dropout(features, keep_prob=0.7, scope='dp2', is_training=self.is_training)
			pred = fully_connected(features, NUM_CLASSES, activation_fn=None, scope='fc3',
			                       is_training=self.is_training)
		
		return pred
	
	def create_loss(self, pred, label):
		""" pred: B * NUM_CLASSES,
			label: B, """
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
		cls_loss = tf.reduce_mean(loss)
		tf.summary.scalar('classification loss', cls_loss)
		
		return cls_loss


if __name__ == '__main__':
	
	batch_size, num_cls = BATCH_SIZE, NUM_CLASSES
	lr_clip, base_lr, lr_decay_steps, lr_decay_rate = 1e-6, 1e-4, 50000, .7
	
	is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
	global_step = tf.Variable(0, trainable=False, name='global_step')
	
	inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
	npts_pl = tf.placeholder(tf.int32, (batch_size,), 'num_points')
	labels_pl = tf.placeholder(tf.int32, (batch_size,), 'ground_truths')
	learning_rate = tf.train.exponential_decay(base_lr, global_step,
	                                           lr_decay_steps, lr_decay_rate,
	                                           staircase=True, name='lr')
	learning_rate = tf.maximum(learning_rate, lr_clip)
	
	# model_module = importlib.import_module('./pcn_cls', './')
	model = Model(inputs_pl, npts_pl, labels_pl, is_training_pl)
	trainer = tf.train.AdamOptimizer(learning_rate)
	train_op = trainer.minimize(model.loss, global_step)
	
	print('\n\n\n==========')
	print('pred', model.pred)
	print('loss', model.loss)
	# seems like different from the what the paper has claimed:
	saver = tf.train.Saver()
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = True
	sess = tf.Session(config=config)
	
	# Init variables
	init = tf.global_variables_initializer()
	sess.run(init, {is_training_pl: True})  # restore will cover the random initialized parameters
	
	for idx, var in enumerate(tf.trainable_variables()):
		print(idx, var)

