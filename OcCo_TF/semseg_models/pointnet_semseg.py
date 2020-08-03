#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import os, sys, tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR))
from utils.tf_util import conv2d, max_pool2d, fully_connected, dropout

NUM_POINT = 4096
BATCH_SIZE = 24
# def placeholder_inputs(batch_size, num_point):
# 	pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
# 	labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
# 	return pointclouds_pl, labels_pl


class Model:
	def __init__(self, inputs, labels, is_training, **kwargs):
		self.bn_decay = None
		self.__dict__.update(kwargs)
		self.is_training = is_training
		self.cat_feats = self.create_encoder(inputs)
		self.pred = self.create_decoder()
		self.loss = self.create_loss(labels)

	def create_encoder(self, inputs):
		input_image = tf.expand_dims(inputs, -1)
		# Conv
		net = conv2d(input_image, 64, [1, 9], padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, scope='conv1', bn_decay=self.bn_decay)
		net = conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, scope='conv2', bn_decay=self.bn_decay)
		net = conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, scope='conv3', bn_decay=self.bn_decay)
		net = conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, scope='conv4', bn_decay=self.bn_decay)
		points_feat1 = conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
		                      bn=True, is_training=self.is_training, scope='conv5', bn_decay=self.bn_decay)
		
		# MaxPooling
		pc_feat1 = max_pool2d(points_feat1, [NUM_POINT, 1], padding='VALID', scope='maxpool')
		
		# Fully Connected Layers
		pc_feat1 = tf.reshape(pc_feat1, [BATCH_SIZE, -1])
		pc_feat1 = fully_connected(pc_feat1, 256, bn=True, is_training=self.is_training,
		                           scope='fc1', bn_decay=self.bn_decay)
		pc_feat1 = fully_connected(pc_feat1, 128, bn=True, is_training=self.is_training,
		                           scope='fc2', bn_decay=self.bn_decay)
		
		# Concat
		pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [BATCH_SIZE, 1, 1, -1]), [1, NUM_POINT, 1, 1])
		points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])
		
		return points_feat1_concat

	def create_decoder(self):
		net = conv2d(self.cat_feats, 512, [1, 1], padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, scope='conv6')
		net = conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
		             bn=True, is_training=self.is_training, scope='conv7')
		net = dropout(net, keep_prob=0.7, is_training=self.is_training, scope='dp1')
		net = conv2d(net, 13, [1, 1], padding='VALID', stride=[1, 1],
		             activation_fn=None, scope='conv8')
		net = tf.squeeze(net, [2])
		return net

	def create_loss(self, labels):
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=labels)
		return tf.reduce_mean(loss)


# def get_model(point_cloud, is_training, bn_decay=None):
# 	""" ConvNet baseline, input is BxNx3 gray image """
# 	# i do remember it is B*N*9, (XYZ, RGB, and normalized location to the room)
#
# 	input_image = tf.expand_dims(point_cloud, -1)
# 	# CONV
# 	net = conv2d(input_image, 64, [1, 9], padding='VALID', stride=[1, 1],
# 	                     bn=True, is_training=is_training, scope='conv1/sem', bn_decay=bn_decay)
# 	net = conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
# 	                     bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
# 	net = conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
# 	                     bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
# 	net = conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
# 	                     bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
# 	points_feat1 = conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
# 	                              bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
#
# 	# MAX
# 	pc_feat1 = max_pool2d(points_feat1, [NUM_POINT, 1], padding='VALID', scope='maxpool1')
#
# 	# FC
# 	pc_feat1 = tf.reshape(pc_feat1, [BATCH_SIZE, -1])
# 	pc_feat1 = fully_connected(pc_feat1, 256, bn=True, is_training=is_training,
# 	                                   scope='fc1', bn_decay=bn_decay)
# 	pc_feat1 = fully_connected(pc_feat1, 128, bn=True, is_training=is_training,
# 	                                   scope='fc2', bn_decay=bn_decay)
# 	# print(pc_feat1)
#
# 	# CONCAT
# 	pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [BATCH_SIZE, 1, 1, -1]), [1, NUM_POINT, 1, 1])
# 	points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])
#
# 	# CONV
# 	net = conv2d(points_feat1_concat, 512, [1, 1], padding='VALID', stride=[1, 1],
# 	                     bn=True, is_training=is_training, scope='conv6')
# 	net = conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
# 	                     bn=True, is_training=is_training, scope='conv7')
# 	net = dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
# 	net = conv2d(net, 13, [1, 1], padding='VALID', stride=[1, 1],
# 	                     activation_fn=None, scope='conv8')
# 	net = tf.squeeze(net, [2])
#
# 	return net
#
#
# def get_loss(pred, label):
# 	""" pred: B,N,13
# 		label: B,N """
# 	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
# 	return tf.reduce_mean(loss)


if __name__ == "__main__":
	print('')
	# with tf.Graph().as_default():
	# 	a = tf.placeholder(tf.float32, shape=(32, 4096, 9))
	# 	# default input shape
	# 	net = get_model(a, tf.constant(True))
	# 	with tf.Session() as sess:
	# 		init = tf.global_variables_initializer()
	# 		sess.run(init)
	# 		start = time.time()
	# 		for i in range(100):
	# 			print(i)
	# 			sess.run(net, feed_dict={a: np.random.rand(32, 4096, 9)})
	# 		print(time.time() - start)
