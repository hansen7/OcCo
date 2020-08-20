#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import sys, tensorflow as tf
sys.path.append('../')
from utils.tf_util import mlp_conv, point_maxpool, point_unpool, fully_connected, dropout
from train_cls import NUM_CLASSES
# NUM_CLASSES = 40


class Model:
    def __init__(self, inputs, npts, labels, is_training, **kwargs):
        self.is_training = is_training
        self.features = self.create_encoder(inputs, npts)
        self.pred = self.create_decoder(self.features)
        self.loss = self.create_loss(self.pred, labels)

    def create_encoder(self, inputs, npts):
        """mini-PointNet encoder"""

        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts)
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

    batch_size, num_cls = 16, NUM_CLASSES
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

