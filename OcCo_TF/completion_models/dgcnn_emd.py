#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

# author: Hanchen Wang

import os, sys, tensorflow as tf
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append('../')
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from utils import tf_util
from utils.transform_nets import input_transform_net_dgcnn
from train_completion import BATCH_SIZE, NUM_POINT

# BATCH_SIZE = 8  # otherwise set to 8
# NUM_POINT = 2048  # 3000


class Model:
    def __init__(self, inputs, npts, gt, alpha, **kwargs):
        self.knn = 20
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
    
    def create_encoder(self, point_cloud, npts):

        point_cloud = tf.reshape(point_cloud, (BATCH_SIZE, NUM_POINT, 3))

        adj_matrix = tf_util.pairwise_distance(point_cloud)
        nn_idx = tf_util.knn(adj_matrix, k=self.knn)
        edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=self.knn)

        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net_dgcnn(edge_feature, self.is_training, self.bn_decay, K=3)

        point_cloud_transformed = tf.matmul(point_cloud, transform)
        adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
        nn_idx = tf_util.knn(adj_matrix, k=self.knn)
        edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=self.knn)

        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='dgcnn1', bn_decay=self.bn_decay)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        net1 = net

        adj_matrix = tf_util.pairwise_distance(net)
        nn_idx = tf_util.knn(adj_matrix, k=self.knn)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=self.knn)

        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='dgcnn2', bn_decay=self.bn_decay)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        net2 = net

        adj_matrix = tf_util.pairwise_distance(net)
        nn_idx = tf_util.knn(adj_matrix, k=self.knn)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=self.knn)

        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='dgcnn3', bn_decay=self.bn_decay)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        net3 = net

        adj_matrix = tf_util.pairwise_distance(net)
        nn_idx = tf_util.knn(adj_matrix, k=self.knn)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=self.knn)

        net = tf_util.conv2d(edge_feature, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='dgcnn4', bn_decay=self.bn_decay)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        net4 = net

        net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='agg', bn_decay=self.bn_decay)

        net = tf.reduce_max(net, axis=1, keep_dims=True)

        features = tf.reshape(net, [BATCH_SIZE, -1])
        return features

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = tf_util.mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])
    
        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])
        
            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])
        
            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])
        
            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)
        
            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])
        
            fine = tf_util.mlp_conv(feat, [512, 512, 3]) + center
        return coarse, fine

    def create_loss(self, gt, alpha):
    
        gt_ds = gt[:, :self.coarse.shape[1], :]
        loss_coarse = tf_util.earth_mover(self.coarse, gt_ds)
        tf_util.add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = tf_util.add_valid_summary('valid/coarse_loss', loss_coarse)
    
        loss_fine = tf_util.chamfer(self.fine, gt)
        tf_util.add_train_summary('train/fine_loss', loss_fine)
        update_fine = tf_util.add_valid_summary('valid/fine_loss', loss_fine)
    
        loss = loss_coarse + alpha * loss_fine
        tf_util.add_train_summary('train/loss', loss)
        update_loss = tf_util.add_valid_summary('valid/loss', loss)
    
        return loss, [update_coarse, update_fine, update_loss]
