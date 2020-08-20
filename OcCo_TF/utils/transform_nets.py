#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import os, sys, numpy as np, tensorflow as tf
import tf_util


def input_transform_net_dgcnn(edge_feature, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
    Return:
        Transformation matrix of size 3xK """

    batch_size = edge_feature.get_shape()[0].value
    num_point = edge_feature.get_shape()[1].value

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=-2, keep_dims=True)

    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        # assert(K==3)
        with tf.device('/cpu:0'):
            weights = tf.get_variable('weights', [256, K * K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [K * K],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform


def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    # print('the input shape for t-net:', point_cloud.get_shape())
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    # point_cloud -> Tensor of (batch size, number of points, 3d coordinates)

    input_image = tf.expand_dims(point_cloud, -1)
    # point_cloud -> (batch size, number of points, 3d coordinates, 1)
    # batch size * height * width * channel

    '''tf_util.conv2d(inputs, num_output_channels, kernel_size, scope, stride=[1, 1], padding='SAME',
                        use_xavier=True, stddev=1e-3, weight_decay=0.0, activation_fn=tf.nn.relu,
                        bn=False, bn_decay=None(default is set to 0.9), is_training=None)'''
    net = tf_util.conv2d(input_image, 64, [1, 3],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)

    # net = mlp_conv(input_image, [64, 128, 1024])
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')
    '''(default stride: (2, 2))'''
    # net = tf.reduce_max(net, axis=1, keep_dims=True, name='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert (K == 3)
        weights = tf.get_variable('weights', [256, 3 * K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3 * K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K * K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K * K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
