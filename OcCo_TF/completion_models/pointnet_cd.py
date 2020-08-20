#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import os, sys, tensorflow as tf
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append('../')
from utils.tf_util import conv2d, mlp, mlp_conv, chamfer, add_valid_summary, add_train_summary, max_pool2d
from utils.transform_nets import input_transform_net, feature_transform_net
from train_completion import BATCH_SIZE, NUM_POINT


class Model:
    def __init__(self, inputs, npts, gt, alpha, **kwargs):
        self.__dict__.update(kwargs)  # batch_decay and is_training
        self.num_output_points = 16384  # 1024 * 16
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features = self.create_encoder(inputs, npts)
        self.coarse, self.fine = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(gt, alpha)
        self.outputs = self.fine
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_encoder(self, inputs, npts):
        # with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
        #     features = mlp_conv(inputs, [128, 256])
        #     features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
        #     features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        # with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
        #     features = mlp_conv(features, [512, 1024])
        #     features = tf.reduce_max(features, axis=1, name='maxpool_1')
        # end_points = {}

        # if DATASET =='modelnet40':
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
        # end_points['transform'] = transform
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
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size),
                               tf.linspace(-0.05, 0.05, self.grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            fine = mlp_conv(feat, [512, 512, 3]) + center
        return coarse, fine

    def create_loss(self, gt, alpha):

        loss_coarse = chamfer(self.coarse, gt)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = chamfer(self.fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss = loss_coarse + alpha * loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]
